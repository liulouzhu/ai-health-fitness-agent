import re
from langchain_core.tools import tool as lc_tool
from pydantic import BaseModel
from agent.llm import get_llm
from agent.state import AgentState
from agent.memory_agent import get_memory_agent
from agent.context_manager import get_context_manager
from tools.search_with_tavily import search_with_tavily
from tools.retriever import get_hybrid_retriever


class WorkoutInfo(BaseModel):
    """运动信息结构"""
    type: str
    duration: float
    calories: float


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
            self.hybrid_retriever.fusion_top_k = top_k
            results = self.hybrid_retriever.retrieve(query)
            return results
        except Exception as e:
            print(f"Qdrant检索失败: {e}")
            return []

    def _is_retrieval_sufficient(self, query: str, retrieved_results: list) -> bool:
        """判断检索内容是否足够"""
        if not retrieved_results:
            return False
        retrieved_content = "\n".join([r.text for r in retrieved_results])
        judge_prompt = (
            f"判断以下检索内容是否足够回答用户的问题。\n\n"
            f"用户问题：{query}\n\n检索内容：{retrieved_content[:2000]}\n\n"
            f"如果检索内容足够回答问题，返回\"足够\"。如果不足，返回\"不足\"。\n只返回\"足够\"或\"不足\"，不要其他文字。"
        )
        response = self.llm.invoke([{"role": "user", "content": judge_prompt}])
        return "足够" in response.content

    def _extract_workout_info(self, user_input: str) -> dict:
        """从用户输入中提取运动信息"""
        try:
            response = self.llm_with_tools.invoke([
                {"role": "user", "content": f"从以下文本中提取运动信息：{user_input}"}
            ])
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                args = tool_call.get("args", {}) or tool_call.get("arguments", {})
                if isinstance(args, str):
                    import json
                    args = json.loads(args)
                return {
                    "type": args.get("type", "未知运动"),
                    "duration": float(args.get("duration", 0) or 0),
                    "calories": float(args.get("calories", 0) or 0)
                }
        except Exception as e:
            print(f"[WorkoutAgent] 解析运动数据失败: {e}")
        return {"type": "未知运动", "duration": 0, "calories": 0}

    def run(self, state: AgentState) -> AgentState:
        """执行健身指导"""
        print(f"[WorkoutAgent] run - 开始健身指导")
        is_user_reporting = state.get("intent") == "workout_report"
        try:
            query = state["input_message"]
            ctx_mgr = get_context_manager()

            # 1. 检索：先从向量数据库
            retrieved_results = self.retrieve_from_qdrant(query)
            retrieved_content = "\n".join([r.text for r in retrieved_results]) if retrieved_results else ""

            # 2. 判断检索是否足够，不足则联网补充
            if not self._is_retrieval_sufficient(query, retrieved_results):
                tavily_content = search_with_tavily(query)
                if tavily_content:
                    retrieved_content = f"{retrieved_content}\n\n--- 网络搜索结果 ---\n{tavily_content}"

            # 3. 通过统一上下文接口获取偏好和 task_context
            preferences = ctx_mgr.get_preferences_str()
            if not preferences:
                preferences = "（暂无偏好记录）"
            task_text = ctx_mgr.format_task_context("workout")

            # 4. 使用 ContextManager 统一构建消息（含 token 预算管理）
            extra_sections = {
                "用户偏好": preferences,
                "用户情况": task_text,
                "参考内容": retrieved_content or "无相关内容",
            }
            messages = ctx_mgr.build_prompt_messages(
                "workout",
                state,
                extra_sections=extra_sections,
            )

            response = self.llm.invoke(messages)
            state["workout_result"] = response.content

            # 尝试提取运动信息
            workout_info = self._extract_workout_info(query)

            # 判断是否是用户主动报告（直接记录）还是询问（需要确认）
            if is_user_reporting and (workout_info.get("duration", 0) > 0 or workout_info.get("calories", 0) > 0):
                self.memory_agent.update_daily_stats("workout", workout_info)
                summary = self.memory_agent.get_daily_summary()
                state["response"] = f"{response.content}\n\n---\n已记录。\n\n{summary}"
                state["pending_stats"] = None
                self.memory_agent.clear_pending_stats()
            elif workout_info.get("duration", 0) > 0 or workout_info.get("calories", 0) > 0:
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

        # 更新对话历史（使用 ContextManager 统一管理滑动窗口）
        ctx_mgr.append_messages(state, state["input_message"], state.get("response", ""))
        return state
