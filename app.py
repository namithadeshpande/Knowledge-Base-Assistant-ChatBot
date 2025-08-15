import os
import streamlit as st
from dotenv import load_dotenv
from rag.config import load_provider_from_env, ProviderSettings
from rag.retriever import get_or_create_vectorstore, get_retriever
from rag.ingest import ingest_files
from rag.confluence_ingest import ingest_confluence_space, ingest_confluence_pages
from rag.chain import build_qa_chain, build_corpus_summarizer
from logging_config import setup_logging

load_dotenv()
logger = setup_logging()

st.set_page_config(page_title="RAG Chatbot • Confluence & Docs", page_icon="💬", layout="wide")

st.title("💬 RAG Chatbot — Confluence & Documents")
st.caption("LangChain • FAISS • OpenAI • Streamlit")

# Sidebar: OpenAI Settings
st.sidebar.header("OpenAI Configuration")

# Add usage warnings and shutdown controls
st.sidebar.info("💡 Remember to shutdown when not in use")

# Shutdown button
if st.sidebar.button("🔴 Shutdown App", type="secondary", help="Click to stop the Streamlit server"):
    st.sidebar.success("Shutting down app...")
    st.sidebar.info("Close this browser tab and stop the terminal process with Ctrl+C")
    st.stop()

st.sidebar.divider()

default_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
default_embed = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

model_name = st.sidebar.text_input("Chat model", value=default_model)
embed_model = st.sidebar.text_input("Embedding model", value=default_embed)

persist_dir = st.sidebar.text_input("Vector store path", value="data/vectorstore")
recreate_index = st.sidebar.checkbox("Recreate index", value=False)
top_k = st.sidebar.slider("Top-K documents", min_value=2, max_value=15, value=5)

# Load OpenAI provider configuration
provider_settings: ProviderSettings = load_provider_from_env("OpenAI", model_name, embed_model)
vs = get_or_create_vectorstore(provider_settings, persist_dir, recreate_index=recreate_index)

# Tabs
tab_ingest, tab_chat, tab_summarize = st.tabs(["📥 Ingest", "🧠 Chat", "📝 Summarize"])

# --- Ingest Tab ---
with tab_ingest:
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDF/DOCX/PPTX files",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True
    )
    if st.button("Ingest Uploaded Files") and uploaded_files:
        with st.spinner("Processing & indexing files..."):
            num_docs = ingest_files(uploaded_files, vs, provider_settings)
        st.success(f"Ingested {num_docs} document chunks into the vector store.")

    st.divider()
    st.subheader("Ingest Confluence")
    st.caption("Provide either a space key (with optional CQL) **or** a list of page URLs/IDs.")
    col1, col2 = st.columns(2)
    with col1:
        space_key = st.text_input("Confluence space key (e.g., ENG)")
        cql = st.text_input("Optional Confluence CQL", value="")
        if st.button("Ingest Confluence Space"):
            with st.spinner("Fetching & indexing Confluence pages..."):
                count = ingest_confluence_space(space_key, cql, vs, provider_settings)
            st.success(f"Ingested {count} Confluence page chunks.")
    with col2:
        page_ids = st.text_area("Confluence page IDs or URLs (one per line)")
        if st.button("Ingest Specific Pages"):
            ids = [s.strip() for s in page_ids.splitlines() if s.strip()]
            with st.spinner("Fetching & indexing specified Confluence pages..."):
                count = ingest_confluence_pages(ids, vs, provider_settings)
            st.success(f"Ingested {count} Confluence page chunks.")

# --- Chat Tab ---
with tab_chat:
    st.subheader("Ask a question about your corpus")
    query = st.text_input("Your question")
    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("Ask") and query:
        retriever = get_retriever(vs, k=top_k)
        chain = build_qa_chain(provider_settings, retriever)
        with st.spinner("Thinking..."):
            result = chain.invoke({"question": query})
        st.session_state.history.append(("user", query))
        st.session_state.history.append(("assistant", result["answer"]))

        # render chat and citations
        for role, msg in st.session_state.history[-6:]:
            if role == "user":
                st.chat_message("user").write(msg)
            else:
                st.chat_message("assistant").write(msg)

        st.markdown("**Sources**")
        for m in result.get("source_documents", []):
            meta = m.metadata
            source = meta.get("source", "unknown")
            page = meta.get("page", "")
            url = meta.get("url", "")
            label = f"- {os.path.basename(source) if source else 'doc'}"
            if page != "":
                label += f" (page {page})"
            if url:
                label += f" — {url}"
            st.markdown(label)

# --- Summarize Tab ---
with tab_summarize:
    st.subheader("Summarize")
    mode = st.radio("Summary mode", ["Summarize a single file", "Summarize the entire corpus"])

    if mode == "Summarize a single file":
        single_file = st.file_uploader("Upload a file to summarize", type=["pdf", "docx", "pptx"], accept_multiple_files=False)
        if st.button("Summarize File") and single_file:
            with st.spinner("Summarizing file..."):
                # Create a temporary, in-memory index for just this file
                tmp_vs = get_or_create_vectorstore(provider_settings, persist_dir=None, recreate_index=True, in_memory=True)
                n = ingest_files([single_file], tmp_vs, provider_settings)
                retriever = get_retriever(tmp_vs, k=top_k)
                summarizer = build_corpus_summarizer(provider_settings, retriever)
                summary = summarizer.invoke({"instruction": "Provide a concise, well-structured summary with key points and action items."})
            st.success("Summary ready")
            st.write(summary["summary"])
    else:
        if st.button("Summarize Corpus"):
            with st.spinner("Summarizing corpus..."):
                retriever = get_retriever(vs, k=top_k)
                summarizer = build_corpus_summarizer(provider_settings, retriever)
                summary = summarizer.invoke({"instruction": "Provide a concise, well-structured summary with key points and action items."})
            st.success("Summary ready")
            st.write(summary["summary"])
