from agent.llm import get_llm, get_embedding_model
from agent.state import AgentState
from config import AgentConfig
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from tools.search_with_tavily import search_with_tavily

WORKOUT_AGENT_PROMPT = """你是一个健身教练专家。请根据以下信息回答用户的问题。

用户问题：{query}

参考内容：
{retrieved_content}

直接回复健身指导内容，不需要额外解释。"""

JUDGE_PROMPT = """判断以下检索内容是否足够回答用户的问题。

用户问题：{query}

检索内容：
{retrieved_content}

如果检索内容足够回答问题，返回"足够"。
如果检索内容不足以回答问题，返回"不足"。

只返回"足够"或"不足"，不要其他文字。"""


class WorkoutAgent:
    def __init__(self):
        self.llm = get_llm()
        self.embed_model = get_embedding_model()
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "fitness_guide"
        self._vector_store = None

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embed_model
            )
        return self._vector_store

    def retrieve_from_qdrant(self, query: str, top_k: int = 3) -> list:
        """从本地向量数据库检索"""
        try:
            index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model
            )
            retriever = index.as_retriever(similarity_top_k=top_k)
            results = retriever.retrieve(query)
            return results
        except Exception as e:
            print(f"Qdrant检索失败: {e}")
            return []


    def is_retrieval_sufficient(self, query: str, retrieved_results: list) -> bool:
        """判断检索内容是否足够"""
        if not retrieved_results:
            return False

        # 拼接检索内容
        retrieved_content = "\n".join([r.text for r in retrieved_results])

        # 使用LLM判断是否足够
        judge_prompt = JUDGE_PROMPT.format(
            query=query,
            retrieved_content=retrieved_content[:2000]  # 限制长度
        )

        response = self.llm.invoke([
            {"role": "user", "content": judge_prompt}
        ])

        return "足够" in response.content

    def run(self, state: AgentState) -> AgentState:
        """执行健身指导"""
        query = state["input_message"]

        # 1. 首先从本地向量数据库检索
        retrieved_results = self.retrieve_from_qdrant(query)
        retrieved_content = "\n".join([r.text for r in retrieved_results]) if retrieved_results else ""

        # 2. 判断检索内容是否足够
        if not self.is_retrieval_sufficient(query, retrieved_results):
            # 3. 不足则使用Tavily搜索
            tavily_content = search_with_tavily(query)
            if tavily_content:
                retrieved_content = f"{retrieved_content}\n\n--- 网络搜索结果 ---\n{tavily_content}"

        # 4. 组装提示并调用LLM
        prompt = WORKOUT_AGENT_PROMPT.format(
            query=query,
            retrieved_content=retrieved_content or "无相关内容"
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]

        response = self.llm.invoke(messages)
        state["workout_result"] = response.content
        state["response"] = response.content
        return state
