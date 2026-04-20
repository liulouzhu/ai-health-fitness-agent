import os
from dotenv import load_dotenv

load_dotenv()


class AgentConfig:
    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    # LLM_MODEL = os.getenv("LLM_MODEL")
    LongCat_BASE_URL = os.getenv("LongCat_BASE_URL")
    LongCat_API_KEY = os.getenv("LongCat_API_KEY")
    LLM_MODEL = os.getenv("LongCat_MODEL")
    VLM_MODEL = os.getenv("VLM_MODEL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    RERANK_MODEL = os.getenv("RERANK_MODEL")

    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # 对话历史配置
    MAX_HISTORY_LENGTH = 20  # 最多保存的对话历史条数
    MAX_HISTORY_DISPLAY = 10  # 显示给 LLM 的最近历史条数（兼容旧代码）

    # 统一上下文窗口限制
    MAX_RECENT_MESSAGES = 10  # state["messages"] 运行时滑动窗口上限
    MAX_GENERAL_HISTORY_MESSAGES = 5  # general intent 注入的最近对话轮数
    MAX_LONGTERM_MEMORY_ITEMS = 3  # 长期记忆摘要注入条数上限
    MAX_CONTEXT_CHARS_PER_SECTION = 2000  # 每段上下文最大字符数（防 LLM 输入溢出）

    # 检索配置
    RETRIEVAL_CONTENT_TRUNCATE = 2000  # 检索内容截断长度

    # Qdrant 配置
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

    # 检索配置
    VECTOR_TOP_K = 20  # 向量检索返回数量
    BM25_TOP_K = 20  # BM25 检索返回数量
    FUSION_TOP_K = 5  # RRF 融合后返回数量
    RERANK_TOP_N = 20  # RRF 后送入 Rerank 的候选数量
    USE_RERANK = True  # 是否启用 Rerank 重排序
    USE_QUERY_REWRITE = False  # 是否启用 Query 改写（默认关闭）

    # ============ Token 预算配置 ============
    # 总 token 预算（留一部分给输出），建议不超过模型 context 的 80%
    MAX_TOTAL_CONTEXT_TOKENS = 6000
    # 各分 section 预算（分配给不同上下文层）
    MAX_SYSTEM_CONTEXT_TOKENS = 500  # system prompt 本身的大小
    MAX_EXTRA_CONTEXT_TOKENS = 300  # _build_extra_context 注入的额外上下文
    MAX_CONVERSATION_WINDOW_TOKENS = 1500  # 对话历史窗口
    MAX_USER_MEMORY_TOKENS = 500  # profile + preferences 文本化后的上限
    MAX_TASK_CONTEXT_TOKENS = 400  # intent-specific 业务上下文
    MAX_RETRIEVED_KNOWLEDGE_TOKENS = 2000  # 检索内容上限
    # 字符 → token 估算比率（中英文混合文本的近似值）
    CHARS_PER_TOKEN_ESTIMATE = 3.5

    # ============ 长期记忆加权选择配置 ============
    LONGTERM_MEMORY_MAX_AGE_DAYS = 180  # 评分衰减的最大天数
    LONGTERM_MEMORY_RECENCY_WEIGHT = 0.35  # 时间新近度权重
    LONGTERM_MEMORY_PREF_WEIGHT = 0.30  # 偏好密度权重
    LONGTERM_MEMORY_FACT_WEIGHT = 0.20  # 事实密度权重
    LONGTERM_MEMORY_TOPIC_WEIGHT = 0.15  # 主题相关度权重

    # 意图 → 相关主题映射（用于 topic_relevance 计算）
    INTENT_TOPIC_MAP = {
        "food": ["饮食", "食物", "营养", "热量", "蛋白质", "餐食", "食谱"],
        "food_report": ["饮食", "食物", "营养", "热量", "蛋白质", "餐食", "食谱"],
        "workout": ["运动", "锻炼", "训练", "健身", "跑步", "力量"],
        "workout_report": ["运动", "锻炼", "训练", "健身", "跑步", "力量"],
        "recipe": ["食谱", "菜谱", "饮食", "食物", "营养", "热量", "蛋白质"],
        "stats_query": ["统计", "数据", "热量", "蛋白质", "消耗", "摄入"],
    }

    # ============ 记忆清理配置 ============
    DAILY_STATS_RETENTION_DAYS = 90  # 每日统计保留天数
    LONGTERM_MEMORY_MAX_ENTRIES = 50  # 长期记忆最大条目数
    LONGTERM_MEMORY_MIN_SCORE = 0.15  # 低于此评分的摘要自动淘汰
    PREFERENCE_INVALIDATION_CONFIDENCE = 0.7  # 偏好失效检测置信度阈值

    # 营养目标默认值
    DEFAULT_TARGET_CALORIES = 2000
    DEFAULT_TARGET_PROTEIN = 100
    # Token 估算器模式："tiktoken"=真实 tokenizer，"chars"=字符数/CHARS_PER_TOKEN_ESTIMATE
    TOKEN_ESTIMATOR_MODE = "auto"  # "auto"=优先 tiktoken，失败则 fallback
