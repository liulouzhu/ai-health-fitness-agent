import re
from agent.llm import get_llm, get_embedding_model
from agent.state import AgentState
from config import AgentConfig
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from tools.search_with_tavily import search_with_tavily
from agent.memory_agent import get_memory_agent

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

EXTRACT_WORKOUT_PROMPT = """从以下用户输入中提取运动信息。

用户输入：{user_input}

请以JSON格式返回：
{{"type": "运动类型", "duration": 数字(分钟), "calories": 数字(消耗热量)}}

如果用户没有提供具体数值，设为0。"""


class WorkoutAgent:
    def __init__(self):
        self.llm = get_llm()
        self.embed_model = get_embedding_model()
        self.memory_agent = get_memory_agent()
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

        retrieved_content = "\n".join([r.text for r in retrieved_results])

        judge_prompt = JUDGE_PROMPT.format(
            query=query,
            retrieved_content=retrieved_content[:AgentConfig.RETRIEVAL_CONTENT_TRUNCATE]
        )

        response = self.llm.invoke([
            {"role": "user", "content": judge_prompt}
        ])

        return "足够" in response.content

    def _extract_workout_info(self, user_input: str) -> dict:
        """从用户输入中提取运动信息"""
        prompt = EXTRACT_WORKOUT_PROMPT.format(user_input=user_input)
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        try:
            import json
            data = json.loads(response.content)
        except:
            data = {"type": "未知运动", "duration": 0, "calories": 0}

        return data

    def run(self, state: AgentState) -> AgentState:
        """执行健身指导"""
        print(f"[WorkoutAgent] run - 开始健身指导")
        try:
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

            # 尝试提取运动信息，设置待确认
            workout_info = self._extract_workout_info(query)
            if workout_info.get("duration", 0) > 0 or workout_info.get("calories", 0) > 0:
                pending = {
                    "type": "workout",
                    "data": workout_info,
                    "response": response.content
                }
                state["pending_stats"] = pending
                self.memory_agent.save_pending_stats(pending)
                state["response"] = f"{response.content}\n\n---\n是否将上述运动计入今日消耗统计？（是/否）"
            else:
                state["response"] = response.content
        except Exception as e:
            print(f"[WorkoutAgent] 错误: {e}")
            state["response"] = "抱歉，健身指导服务暂时不可用，请稍后重试。"
            state["workout_result"] = None
            state["pending_stats"] = None
        return state
