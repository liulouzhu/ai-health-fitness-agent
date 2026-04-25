"""
运动咨询 Agent

职责：
- 回答用户的运动/健身问题
- 检索相关知识（Qdrant + Tavily）
- 生成专业的健身建议

只处理 workout 意图（咨询），不处理运动记录。
"""

from agent.llm import get_llm
from agent.state import AgentState
from agent.context_manager import get_context_manager, is_retrieval_sufficient as check_retrieval
from tools.search_with_tavily import search_with_tavily
from tools.retriever import get_hybrid_retriever


class WorkoutAdviceAgent:
    def __init__(self):
        self.llm = get_llm()
        self.collection_name = "fitness_guide"
        self._hybrid_retriever = None

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
            print(f"[WorkoutAdviceAgent] Qdrant检索失败: {e}")
            return []

    def run(
        self,
        state: AgentState,
        extra_sections: dict = None,
        branch_input: str = None,
        append_history: bool = False,
    ) -> AgentState:
        """回答运动咨询问题

        流程：
        1. 检索相关知识
        2. LLM 生成建议
        """
        print(f"[WorkoutAdviceAgent] run - 回答运动咨询")

        query = branch_input or state["input_message"]

        try:
            ctx_mgr = get_context_manager()

            # === Step 1: 检索 ===
            retrieved_results = self.retrieve_from_qdrant(query)
            retrieved_content = "\n".join([r.text for r in retrieved_results]) if retrieved_results else ""

            if not check_retrieval(retrieved_content, query=query, domain="workout"):
                tavily_content = search_with_tavily(query)
                if tavily_content:
                    retrieved_content = f"{retrieved_content}\n\n--- 网络搜索结果 ---\n{tavily_content}"

            # === Step 2: LLM 生成建议 ===
            if extra_sections is not None:
                extra_sections = dict(extra_sections)
                extra_sections["参考内容"] = retrieved_content or "无相关内容"
            else:
                preferences = ctx_mgr.get_preferences_str() or "（暂无偏好记录）"
                task_text = ctx_mgr.format_task_context("workout")
                extra_sections = {
                    "用户偏好": preferences,
                    "用户情况": task_text,
                    "参考内容": retrieved_content or "无相关内容",
                }

            messages = ctx_mgr.build_prompt_messages(
                "workout",
                state,
                extra_sections=extra_sections,
                user_input=query,
            )
            response = self.llm.invoke(messages)

            state["workout_result"] = response.content
            state["requires_confirmation"] = False

        except Exception as e:
            print(f"[WorkoutAdviceAgent] 错误: {e}")
            state["response"] = "抱歉，健身指导服务暂时不可用，请稍后重试。"
            state["workout_result"] = None
            state["requires_confirmation"] = False

        return state
