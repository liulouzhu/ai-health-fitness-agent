from langchain_openai import ChatOpenAI
from config import AgentConfig
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeBatchTextEmbeddingModels, DashScopeTextEmbeddingType

# 全局单例实例
_llm_instance = None
_vlm_instance = None
_embedding_model_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model=AgentConfig.LLM_MODEL,
            api_key=AgentConfig.LongCat_API_KEY,
            base_url=AgentConfig.LongCat_BASE_URL,
        )
    return _llm_instance

def get_vlm():
    global _vlm_instance
    if _vlm_instance is None:
        _vlm_instance = ChatOpenAI(
            model=AgentConfig.VLM_MODEL,
            api_key=AgentConfig.LLM_API_KEY,
            base_url=AgentConfig.LLM_BASE_URL,
        )
    return _vlm_instance

def get_embedding_model():
    global _embedding_model_instance
    if _embedding_model_instance is None:
        _embedding_model_instance = DashScopeEmbedding(
            model=AgentConfig.EMBEDDING_MODEL or "text-embedding-v3",
            api_key=AgentConfig.LLM_API_KEY
        )
    return _embedding_model_instance