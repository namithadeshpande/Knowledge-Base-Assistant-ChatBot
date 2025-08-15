import os
from typing import List
from langchain_community.document_loaders.confluence import ConfluenceLoader
from langchain.schema import Document
from .retriever import make_splitter, save_vectorstore

def _get_confluence_loader():
    url = os.getenv("CONFLUENCE_URL")
    username = os.getenv("CONFLUENCE_USERNAME")
    token = os.getenv("CONFLUENCE_API_TOKEN")
    if not (url and username and token):
        raise ValueError("Set CONFLUENCE_URL, CONFLUENCE_USERNAME, and CONFLUENCE_API_TOKEN in your environment.")
    return ConfluenceLoader(
        url=url,
        username=username,
        api_key=token,
    )

def ingest_confluence_space(space_key: str, cql: str, vectorstore, provider_settings) -> int:
    loader = _get_confluence_loader()
    docs = loader.load(space_key=space_key, cql=cql or None)
    for d in docs:
        d.metadata["url"] = d.metadata.get("source", "")
    return _add_docs(docs, vectorstore)

def ingest_confluence_pages(ids_or_urls: List[str], vectorstore, provider_settings) -> int:
    loader = _get_confluence_loader()
    docs = loader.load(page_ids=ids_or_urls)
    for d in docs:
        d.metadata["url"] = d.metadata.get("source", "")
    return _add_docs(docs, vectorstore)

def _add_docs(docs: List[Document], vectorstore) -> int:
    splitter = make_splitter()
    chunks = splitter.split_documents(docs)
    if len(chunks) == 0:
        return 0
    vectorstore.add_documents(chunks)
    try:
        save_vectorstore(vectorstore, "data/vectorstore")
    except Exception:
        pass
    return len(chunks)
