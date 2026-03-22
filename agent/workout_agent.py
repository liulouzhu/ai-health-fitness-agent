import re
from langchain_core.tools import tool
from pydantic import BaseModel
from agent.llm import get_llm
from agent.state import AgentState
from config import AgentConfig
from tools.search_with_tavily import search_with_tavily
from agent.memory_agent import get_memory_agent
from tools.retriever import get_hybrid_retriever

WORKOUT_AGENT_PROMPT = """你是一个健身教练专家。请根据以下信息回答用户的问题。

用户偏好（请在推荐时注意）：
{preferences}

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


class WorkoutInfo(BaseModel):
    """运动信息结构"""
    type: str
    duration: float
    calories: float


def _extract_workout_info(self, user_input: str) -> dict:
    """从用户输入中提取运动信息"""
    try:
        response = self.llm_with_tools.invoke([
            {"role": "user", "content": f"从以下文本中提取运动信息：{user_input}"}
        ])

        if response.tool_calls:
            # 直接从 tool_call 的 arguments 解析
            tool_call = response.tool_calls[0]
            args = tool_call.get("args", {}) or tool_call.get("arguments", {})
            if isinstance(args, str):
                import json
                args = json.loads(args)
            return {
                "type": args.get("type", "未知运动"),
                "duration": float(args.get("duration", 0)),
                "calories": float(args.get("calories", 0))
            }
    except Exception as e:
        print(f"[WorkoutAgent] 解析运动数据失败: {e}")

    return {"type": "未知运动", "duration": 0, "calories": 0}


class WorkoutAgent:
    def __init__(self):
        self.llm = get_llm()
        self.memory_agent = get_memory_agent()
        self.collection_name = "fitness_guide"
        self._hybrid_retriever = None
        self.llm_with_tools = self.llm.bind_tools([WorkoutInfo])

    @property
    def hybrid_retriever(self):
        if self._hybrid_retriever is None:
            self._hybrid_retriever = get_hybrid_retriever(self.collection_name)
        return self._hybrid_retriever

    def retrieve_from_qdrant(self, query: str, top_k: int = 3) -> list:
        """从本地向量数据库检索（混合搜索：向量 + BM25）"""
        try:
            # 使用混合检索器，检索更多结果再截取
            self.hybrid_retriever.fusion_top_k = top_k
            results = self.hybrid_retriever.retrieve(query)
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
        try:
            response = self.llm_with_tools.invoke([
                {"role": "user", "content": f"从以下文本中提取运动信息：{user_input}"}
            ])

            if response.tool_calls:
                parsed = response.tool_calls[0].parsed
                return {
                    "type": parsed.type,
                    "duration": parsed.duration,
                    "calories": parsed.calories
                }
        except Exception as e:
            print(f"[WorkoutAgent] 解析运动数据失败: {e}")

        return {"type": "未知运动", "duration": 0, "calories": 0}

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

            # 4. 获取用户偏好
            preferences = self.memory_agent.get_preferences_for_context()
            if not preferences:
                preferences = "（暂无偏好记录）"

            # 5. 组装提示并调用LLM
            prompt = WORKOUT_AGENT_PROMPT.format(
                query=query,
                retrieved_content=retrieved_content or "无相关内容",
                preferences=preferences
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
                state["pending_stats"] = {
                    "type": "workout",
                    "data": workout_info,
                    "response": response.content
                }
                self.memory_agent.save_pending_stats(state["pending_stats"])
                state["response"] = f"{response.content}\n\n---\n是否将上述运动计入今日消耗统计？（是/否）"
            else:
                state["response"] = response.content
        except Exception as e:
            print(f"[WorkoutAgent] 错误: {e}")
            state["response"] = "抱歉，健身指导服务暂时不可用，请稍后重试。"
            state["workout_result"] = None
            state["pending_stats"] = None
        return state
