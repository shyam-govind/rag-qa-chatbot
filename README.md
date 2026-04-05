Advanced RAG Q&A; Chatbot

Overview
A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF/TXT files and ask
Key Features
- Upload multiple PDF/TXT files
- Semantic search using embeddings
- Local LLM inference via Ollama
- Chat history tracking
- Source document display for transparency

Architecture
1. File Upload (Streamlit)
2. Document Loading (TextLoader / PyPDFLoader)
3. Text Splitting (CharacterTextSplitter)
4. Embeddings (all-MiniLM-L6-v2)
5. Vector Store (FAISS)
6. Retrieval (Top-k search)
7. LLM (Ollama - phi)
8. Response Generation

Tech Stack
Python, Streamlit, LangChain, FAISS, HuggingFace Embeddings, Ollama (Local LLM)

Setup Instructions
1. Clone repository
2. Create virtual environment
3. Install dependencies
4. Install and run Ollama
5. Run Streamlit app

Run Commands
pip install -r requirements.txt
streamlit run app.py

Author
Shyam Govind
