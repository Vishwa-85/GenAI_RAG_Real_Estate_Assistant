# Standard library imports
from uuid import uuid4  # Generates unique IDs for each document chunk in the vector store
from pathlib import Path  # Cross-platform path handling

# Third-party imports
from dotenv import load_dotenv  # Loads API keys and config from a .env file

# LangChain components for building the RAG pipeline
from langchain_community.document_loaders import UnstructuredURLLoader  # Scrapes and parses raw content from URLs
from langchain_text_splitters import RecursiveCharacterTextSplitter     # Splits long documents into overlapping chunks
from langchain_classic.chains import RetrievalQAWithSourcesChain        # QA chain that retrieves context and cites sources
from langchain_chroma import Chroma                                      # Chroma vector store integration for LangChain
from langchain_groq import ChatGroq                                      # Groq-hosted LLM (LLaMA 3.3) client
from langchain_huggingface import HuggingFaceEmbeddings                 # Wraps HuggingFace embedding models for LangChain
from sentence_transformers import SentenceTransformer                   # Underlying library for the embedding model

# Load environment variables (e.g., GROQ_API_KEY) from the .env file in the project root
load_dotenv()

# --- Configuration Constants ---
CHUNK_SIZE = 1000                                                   # Max number of characters per document chunk
CHUNK_OVERLAP = 200                                                 # Characters shared between consecutive chunks to preserve context
COLLECTION_NAME = "real_estate"                                     # Logical namespace for this dataset inside Chroma
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"          # Lightweight, fast embedding model (384-dim vectors)
VECTOR_STORE_DIR = Path(__file__).parent / "resources/vectorstore"  # Absolute path to persist Chroma's SQLite + index files


# Both are set to None at startup and initialized on first use via initialize_components(),
# avoiding unnecessary model loading if only parts of the module are imported.
llm = None
vector_store = None


def initialize_components():
    """
    Initializes the LLM and the Chroma vector store (only once per process).
    Uses global variables so the heavy models are loaded at most once across all calls.
    """
    global llm, vector_store

    # Only instantiate the LLM if it hasn't been created yet
    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",            # 70B-parameter LLaMA 3.3 model served via Groq's API
            temperature=0.2,                            # Higher temperature → more varied/creative answers
            max_tokens=600                              # Cap output length to control latency and cost
        )
    
    if vector_store is None:
        # Initialize the HuggingFace embedding function used to encode text into vectors
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}  # L2-normalize vectors → cosine similarity == dot product
        )
        
        # Connect to (or create) the Chroma collection at the specified directory
        # Chroma will persist data to disk so the index survives between runs
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=ef
        )


def process_urls(urls):
    """
    Full ingestion pipeline: scrape → split → embed → store.

    Steps:
      1. Resets the existing collection so stale data is not mixed with fresh content.
      2. Loads raw text from each URL using UnstructuredURLLoader.
      3. Splits documents into small, overlapping chunks for fine-grained retrieval.
      4. Truncates each chunk to 1 000 characters as a safety cap before embedding.
      5. Assigns a unique UUID to each chunk and upserts them into Chroma.

    Args:
        urls (list[str]): Public web page URLs to scrape and index.
    """
    yield "Initializing components"
    initialize_components()

    # Clear all previously stored vectors so results reflect only the current batch of URLs
    yield "Resetting vector store"
    vector_store.reset_collection()

    yield "Load data"
    # UnstructuredURLLoader fetches each URL and uses the `unstructured` library to
    # extract clean text (strips HTML tags, navigation menus, etc.)
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()  # Returns a list of LangChain Document objects

    yield "Split text"
    # RecursiveCharacterTextSplitter tries to split on paragraph breaks first (\n\n),
    # then newlines, then sentences, then spaces — preserving as much semantic coherence
    # as possible while staying within CHUNK_SIZE.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    docs = text_splitter.split_documents(data)

    # Hard-cap each chunk's content at 1 000 characters.
    # Prevents unexpectedly large chunks (e.g., minified JS or data blobs that slip through)
    # from hitting the embedding model's token limit.
    for doc in docs:
        doc.page_content = doc.page_content[:1000]

    print(f"Add {len(docs)} docs to vector db")
    # Generate a unique ID per chunk; Chroma requires explicit IDs when adding documents
    yield "Adding chunks to vector database"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)


def generate_answer(query):
    """
    Retrieves relevant chunks from the vector store and generates a grounded answer.

    Uses RetrievalQAWithSourcesChain, which:
      - Embeds the query and performs similarity search against stored chunks.
      - Feeds the top-k chunks as context to the LLM.
      - Returns both the answer text and the source URLs it drew from.

    Args:
        query (str): Natural language question from the user.

    Returns:
        tuple[str, str]: (answer, sources) where `sources` is a comma-separated
                         string of URLs cited by the chain.

    Raises:
        RuntimeError: If process_urls() has not been called yet (vector store is empty/None).
    """
    if vector_store is None:
        raise RuntimeError("Vector DB is not initialised")
    
    # Build the retrieval-augmented QA chain; as_retriever() uses cosine similarity by default
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever()
    )

    # Invoke the chain; the result dict always contains "answer" and "sources" keys
    result = chain.invoke({"question": query}, return_only_output=True)

    # "sources" may be an empty string if the LLM couldn't attribute the answer to any chunk
    sources = result.get("sources", "")

    return result['answer'], sources


# --- Quick smoke-test when the module is run directly ---
if __name__ == '__main__':
    urls = [
        "https://techcrunch.com/",
        "https://venturebeat.com/ai",
        "https://www.reuters.com/technology/"
    ]
    process_urls(urls)
    answer, sources = generate_answer("What are the top headlines today?")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")