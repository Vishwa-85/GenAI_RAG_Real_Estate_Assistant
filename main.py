import streamlit as st
from rag import process_urls, generate_answer

st.title("Real Estate Research Assistant")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

placeholder = st.empty()
status_placeholder = st.empty()
process_url_button = st.sidebar.button("Process URLs")

if process_url_button:
    urls = [url for url in (url1, url2, url3) if url != '']
    if len(urls) == 0:
        st.warning("Please provide at least one valid URL")
    else:
        for status in process_urls(urls):
            status_placeholder.text(status)

query = st.text_input("Enter your question: ")
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer: ")
        st.write(answer)

        if sources:
            st.subheader("Sources: ")
            for source in sources.split(","):
                st.write(source.strip())
    except RuntimeError as e:
        st.error(str(e))
        placeholder.text("Please process urls first")