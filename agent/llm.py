from langchain_openai import ChatOpenAI
from config import AgentConfig
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeBatchTextEmbeddingModels, DashScopeTextEmbeddingType

def get_llm():
    llm = ChatOpenAI(
        model=AgentConfig.LLM_MODEL,
        api_key=AgentConfig.LLM_API_KEY,
        base_url=AgentConfig.LLM_BASE_URL,
    )
    return llm

def get_vlm():
    vlm = ChatOpenAI(
        model=AgentConfig.VLM_MODEL,
        api_key=AgentConfig.LLM_API_KEY,
        base_url=AgentConfig.LLM_BASE_URL,
    )
    return vlm

def get_embedding_model():
    embedding_model = DashScopeEmbedding(
        model=AgentConfig.EMBEDDING_MODEL or "text-embedding-v3",
        api_key=AgentConfig.LLM_API_KEY
    )
    return embedding_model