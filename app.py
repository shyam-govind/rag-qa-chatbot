import streamlit as st
import tempfile

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# CACHE ONLY EMBEDDINGS (for faster retrieval)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# LOAD FILES
def load_files(uploaded_files):
    docs = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        if file.name.endswith(".txt"):
            loader = TextLoader(tmp_path)
        elif file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            continue

        docs.extend(loader.load())

    return docs

# RAG PIPELINE (NO CACHE HERE)
def create_qa(docs):

    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    texts = text_splitter.split_documents(docs)

    embeddings = load_embeddings()

    db = FAISS.from_documents(texts, embeddings)

    llm = Ollama(
        model="phi",        
        num_predict=100
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return qa

# UI
st.set_page_config(page_title="Advanced RAG Chatbot")
st.title("RAG implemented Q/A Chatbot")

uploaded_files = st.file_uploader(
    "Upload TXT or PDF files",
    accept_multiple_files=True
)

# CHAT HISTORY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# BUILD PIPELINE ONLY WHEN FILES CHANGE
if uploaded_files:
    documents = load_files(uploaded_files)
    qa = create_qa(documents)

    query = st.text_input("Ask a question:")

    if query:
        result = qa.invoke(query)

        answer = result["result"]
        sources = result["source_documents"]

        st.session_state.chat_history.append((query, answer, sources))

# DISPLAY CHAT HISTORY
for q, a, s in st.session_state.chat_history[::-1]:

    st.markdown(f"**Question:** {q}")
    st.markdown(f"**Answer:** {a}")

    with st.expander("📄 Sources"):
        for doc in s:
            st.write(doc.page_content[:300])

    st.markdown("---")