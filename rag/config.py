from dataclasses import dataclass
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings

@dataclass
class ProviderSettings:
    provider: str
    chat_model: str
    embed_model: str
    llm: object
    embedder: object

def _init_openai(chat_model: str, embed_model: str) -> ProviderSettings:
    llm = ChatOpenAI(model=chat_model, temperature=0)
    embedder = OpenAIEmbeddings(model=embed_model)
    return ProviderSettings("OpenAI", chat_model, embed_model, llm, embedder)

def _init_azure(chat_model: str, embed_model: str) -> ProviderSettings:
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    llm = AzureChatOpenAI(azure_deployment=chat_model, api_version=api_version, temperature=0)
    embedder = AzureOpenAIEmbeddings(azure_deployment=embed_model, api_version=api_version)
    return ProviderSettings("Azure OpenAI", chat_model, embed_model, llm, embedder)

def load_provider_from_env(provider: str, chat_model: str, embed_model: str) -> ProviderSettings:
    if provider == "Azure OpenAI":
        return _init_azure(chat_model, embed_model)
    return _init_openai(chat_model, embed_model)
