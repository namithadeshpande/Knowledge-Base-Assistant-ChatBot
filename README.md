# RAG Chatbot for Confluence & Documents (Streamlit + LangChain + FAISS)

A production-grade chatbot that explains/summarizes **Confluence pages**, **PDF**, **DOCX**, and **PPTX** documents using **LangChain**, **FAISS**, and **OpenAI or Azure OpenAI**. Frontend is built with **Streamlit**.

## Features
- Upload **PDF/DOCX/PPTX** or ingest **Confluence** pages.
- Persisted **FAISS** vector store on disk.
- Choice of **OpenAI** or **Azure OpenAI** for both **LLM** and **embeddings**.
- **Citations** with source metadata (filename, page, url).
- **Summarize** any single file or the entire indexed corpus.
- Simple, production-friendly structure with logging and modular code.

## Quickstart

### 1) Python environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Environment variables
Copy `.env.example` to `.env` and fill in at least one provider.

#### If using OpenAI:
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini     # or gpt-4o, gpt-4.1, etc.
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

#### If using Azure OpenAI:
```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=your-gpt4o-deployment
AZURE_OPENAI_EMBED_DEPLOYMENT=your-embed-deployment
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

> You can also set these via your shell or a secret manager.

### 3) Run the app
```bash
streamlit run app.py
```

### 4) Use the app
- In the sidebar, choose your provider (OpenAI or Azure OpenAI).
- **Upload** PDFs/DOCX/PPTX or enter **Confluence** details to ingest.
- Ask questions, get **grounded** answers with **citations**.
- Use **Summarize** to get document or corpus summaries.

## Project Structure
```
.
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   └── vectorstore/              # persisted FAISS index
├── rag/
│   ├── chain.py
│   ├── config.py
│   ├── ingest.py
│   ├── retriever.py
│   └── confluence_ingest.py
└── utils/
    └── logging_config.py
```

## Confluence notes
- This project uses `langchain-community`'s `ConfluenceLoader`. You will need:
  - `CONFLUENCE_USERNAME` and a **Confluence API token** as `CONFLUENCE_API_TOKEN`.
  - `CONFLUENCE_URL` (base URL, e.g. `https://yourdomain.atlassian.net/wiki`).
- In the UI, supply **space key** and an optional **CQL** to filter pages (or a list of page IDs/URLs).

## License
MIT
