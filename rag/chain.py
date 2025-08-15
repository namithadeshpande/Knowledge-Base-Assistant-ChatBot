from typing import Dict, Any, List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the provided context.\n"
    "- If the answer cannot be found in the context, say you don't know.\n"
    "- Cite the most relevant sources at the end if available.\n"
    "Keep answers concise but accurate."
)

QA_PROMPT = PromptTemplate.from_template(
    "{system}\nQuestion: {question}\nContext:\n{context}\nAnswer:"
)

def _format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "")
        page = meta.get("page", "")
        url = meta.get("url", "")
        header = f"[source: {src}{' p.'+str(page) if page!='' else ''}{' | '+url if url else ''}]"
        parts.append(header + "\n" + d.page_content)
    return "\n\n---\n\n".join(parts)

def build_qa_chain(provider_settings, retriever):
    llm = provider_settings.llm
    prompt = QA_PROMPT.partial(system=SYSTEM_PROMPT)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    class _Wrapper:
        def invoke(self, inputs: Dict[str, Any]):
            question = inputs["question"]
            docs = retriever.get_relevant_documents(question)
            answer = chain.invoke(question)
            return {"answer": answer, "source_documents": docs}

    return _Wrapper()

SUMMARY_PROMPT = PromptTemplate.from_template(
    "You are a senior technical writer. Using the retrieved context, write a concise, well-structured summary.\n"
    "Include:\n"
    "- Executive summary (3-5 bullets)\n"
    "- Key details\n"
    "- Risks/unknowns\n"
    "- Action items (if any)\n"
    "Instruction: {instruction}\n\n"
    "Context:\n"
    "{context}\n\n"
    "Summary:"
)

def build_corpus_summarizer(provider_settings, retriever):
    llm = provider_settings.llm
    prompt = SUMMARY_PROMPT

    chain = (
        {"context": retriever | _format_docs, "instruction": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    class _Wrapper:
        def invoke(self, inputs: Dict[str, Any]):
            instruction = inputs.get("instruction", "")
            docs = retriever.get_relevant_documents("Comprehensive overview of the corpus")
            summary = chain.invoke(instruction)
            return {"summary": summary, "source_documents": docs}

    return _Wrapper()
