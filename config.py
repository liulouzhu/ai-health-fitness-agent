import os
from dotenv import load_dotenv
load_dotenv()

class AgentConfig:

    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL")
    VLM_MODEL = os.getenv("VLM_MODEL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # 对话历史配置
    MAX_HISTORY_LENGTH = 20       # 最多保存的对话历史条数
    MAX_HISTORY_DISPLAY = 10      # 显示给 LLM 的最近历史条数

    # 检索配置
    RETRIEVAL_CONTENT_TRUNCATE = 2000  # 检索内容截断长度

    # Qdrant 配置
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

    # 检索配置
    VECTOR_TOP_K = 20           # 向量检索返回数量
    BM25_TOP_K = 20             # BM25 检索返回数量
    FUSION_TOP_K = 5            # 融合后返回数量
    USE_QUERY_REWRITE = True    # 是否启用 Query 改写