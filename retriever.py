import os
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_or_create_vectorstore(provider_settings, persist_dir: Optional[str], recreate_index: bool=False, in_memory: bool=False):
    embeddings = provider_settings.embedder

    if in_memory:
        return FAISS.from_texts(["Initialization vector"], embedding=embeddings)

    if persist_dir is None:
        raise ValueError("persist_dir must be provided unless in_memory=True")

    index_path = os.path.join(persist_dir)
    os.makedirs(index_path, exist_ok=True)

    if recreate_index:
        return FAISS.from_texts(["Initialization vector"], embedding=embeddings)

    try:
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return FAISS.from_texts(["Initialization vector"], embedding=embeddings)

def save_vectorstore(vs, persist_dir: str):
    os.makedirs(persist_dir, exist_ok=True)
    vs.save_local(persist_dir)

def make_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )

def get_retriever(vs, k: int = 5):
    return vs.as_retriever(search_kwargs={"k": k})
