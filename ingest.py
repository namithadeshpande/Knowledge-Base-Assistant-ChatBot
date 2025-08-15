import os
from typing import List, Iterable
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.schema import Document
from .retriever import make_splitter, save_vectorstore

def _load_one_file(fpath: str):
    ext = os.path.splitext(fpath.lower())[1]
    if ext == ".pdf":
        loader = PyPDFLoader(fpath)
        docs = loader.load()
        for d in docs:
            d.metadata["page"] = d.metadata.get("page", "")
        return docs
    if ext == ".docx":
        loader = Docx2txtLoader(fpath)
        return loader.load()
    if ext == ".pptx":
        loader = UnstructuredPowerPointLoader(fpath)
        return loader.load()
    raise ValueError(f"Unsupported file type: {ext}")

def ingest_files(uploaded_files, vectorstore, provider_settings) -> int:
    import tempfile
    tempdir = tempfile.mkdtemp()
    paths = []
    for uf in uploaded_files:
        path = os.path.join(tempdir, uf.name)
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
        paths.append(path)

    all_docs: List[Document] = []
    for p in paths:
        docs = list(_load_one_file(p))
        for d in docs:
            d.metadata["source"] = p
        all_docs.extend(docs)

    splitter = make_splitter()
    chunks = splitter.split_documents(all_docs)

    if len(chunks) == 0:
        return 0

    vectorstore.add_documents(chunks)

    try:
        save_vectorstore(vectorstore, "data/vectorstore")
    except Exception:
        pass

    return len(chunks)
