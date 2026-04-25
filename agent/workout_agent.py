"""
运动 Agent

职责（graph-native 改造后）：
- 输入解析与运动信息提取
- 生成候选记录（写入 pending_confirmation.candidate_workout）
- 设置 pending_confirmation / requires_confirmation
- 由 graph 的 confirm_node → commit_node 完成实际写入
"""

from pydantic import BaseModel
from agent.llm import get_llm
from agent.state import AgentState
from agent.memory import get_memory_agent
from agent.context_manager import get_context_manager, is_retrieval_sufficient as check_retrieval
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
            print(f"[WorkoutAgent] Qdrant检索失败: {e}")
            return []

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

    def run(
        self,
        state: AgentState,
        extra_sections: dict = None,
        branch_input: str = None,
        append_history: bool = True,
    ) -> AgentState:
        """执行健身指导（graph-native 版本）

        流程：
        1. 检索
        2. LLM 生成指导
        3. 提取运动信息
        4. 生成候选记录（写入 pending_confirmation.candidate_workout）
        5. 设置 pending_confirmation / requires_confirmation
        6. 写入 workout_result + final_response

        实际 commit 由 graph 的 commit_node 负责。

        Args:
            state: LangGraph AgentState
            extra_sections: 可选，由 planner 预生成的分支上下文。如果提供，优先使用。
            branch_input: 可选，由 planner 预生成的分支专属输入片段。如果提供，优先使用。
        """
        source_intents = state.get("source_intents") or []
        is_user_reporting = (
            "workout_report" in source_intents
            or state.get("intent") == "workout_report"
        )
        has_advice_intent = "workout" in source_intents
        is_multi_intent = is_user_reporting and has_advice_intent
        print(f"[WorkoutAgent] run - is_user_reporting={is_user_reporting}, has_advice_intent={has_advice_intent}, is_multi_intent={is_multi_intent}, intent={state.get('intent')}, source_intents={source_intents}")

        # 优先使用原始输入（多意图场景需要完整输入），否则用分支专属输入
        if is_multi_intent:
            query = state["input_message"]
            print(f"[WorkoutAgent] run - 多意图，使用原始输入: {query[:50]}...")
        elif branch_input:
            query = branch_input
            print(f"[WorkoutAgent] run - 使用分支专属输入: {query[:50]}...")
        else:
            query = state["input_message"]
            print(f"[WorkoutAgent] run - 使用原始输入: {query[:50]}...")

        try:
            ctx_mgr = get_context_manager()

            # === Step 1: 检索 ===
            # 多意图时对问题部分也做检索
            retrieval_query = query
            if is_multi_intent:
                # 提取问题部分用于检索（如"背部应该怎么练"）
                import re
                clauses = re.split(r"[；;。！？!?\n]+", state["input_message"])
                question_parts = [c.strip() for c in clauses if "怎么" in c or "如何" in c or "？" in c or "?" in c]
                if question_parts:
                    retrieval_query = " ".join(question_parts)
                    print(f"[WorkoutAgent] run - 多意图检索 query: {retrieval_query[:50]}...")

            retrieved_results = self.retrieve_from_qdrant(retrieval_query)
            retrieved_content = "\n".join([r.text for r in retrieved_results]) if retrieved_results else ""

            if not check_retrieval(retrieved_content, query=retrieval_query, domain="workout"):
                tavily_content = search_with_tavily(retrieval_query)
                if tavily_content:
                    retrieved_content = f"{retrieved_content}\n\n--- 网络搜索结果 ---\n{tavily_content}"

            # === Step 2: LLM 生成 ===
            # 如果有预生成的 extra_sections，优先使用；否则自己构建
            if extra_sections is not None:
                print(f"[WorkoutAgent] run - 使用预生成的 extra_sections")
                # 合并预生成的 sections 和检索内容
                extra_sections = dict(extra_sections)
                extra_sections["参考内容"] = retrieved_content or "无相关内容"
            else:
                preferences = ctx_mgr.get_preferences_str()
                if not preferences:
                    preferences = "（暂无偏好记录）"
                task_text = ctx_mgr.format_task_context("workout")

                extra_sections = {
                    "用户偏好": preferences,
                    "用户情况": task_text,
                    "参考内容": retrieved_content or "无相关内容",
                }

            # 多意图时注入明确指令，让 LLM 同时处理记录和回答
            if is_multi_intent:
                extra_sections["⚠️ 必须完成的两个任务（缺一不可）"] = (
                    "用户输入包含两个意图，你必须同时处理：\n\n"
                    "【任务1 - 运动记录】确认用户的运动数据（类型、时长、预估消耗）\n\n"
                    "【任务2 - 运动咨询】回答用户提出的健身问题，给出具体动作、组数、次数等专业建议\n\n"
                    "⚠️ 你的回复必须包含以上两个任务的内容，只完成其中一个是不合格的！"
                )

            messages = ctx_mgr.build_prompt_messages(
                "workout",
                state,
                extra_sections=extra_sections,
            )

            # 多意图时：在系统消息后插入强制指令，确保 LLM 不会忽略
            if is_multi_intent:
                messages.insert(1, {
                    "role": "system",
                    "content": (
                        "⚠️ 重要：用户的输入包含两个意图——运动记录 + 运动咨询。"
                        "你必须在回复中同时处理这两个意图！\n"
                        "1. 先总结用户的运动记录\n"
                        "2. 再详细回答用户的健身问题（给出具体动作、组数、次数建议）\n"
                        "只处理其中一个意图是不合格的回复！"
                    )
                })
            response = self.llm.invoke(messages)
            state["workout_result"] = response.content

            # === Step 3: 提取运动数据 ===
            workout_info = self._extract_workout_info(query)
            # === Step 4: 设置确认态 ===
            # confirmed=True → commit_node 直接 commit（workout_report）
            # confirmed=None → confirm_node 询问用户
            if is_user_reporting and (workout_info.get("duration", 0) > 0 or workout_info.get("calories", 0) > 0):
                state["requires_confirmation"] = True
                state["pending_confirmation"] = {
                    "action": "log_workout",
                    "candidate_meal": None,
                    "candidate_workout": workout_info,
                    "analysis_text": response.content,
                    "confirmed": True,
                }
                state["final_response"] = (
                    f"{response.content}\n\n---\n"
                    f"（已提交，等待确认...）"
                )
                state["response"] = state["final_response"]
            elif workout_info.get("duration", 0) > 0 or workout_info.get("calories", 0) > 0:
                state["requires_confirmation"] = True
                state["pending_confirmation"] = {
                    "action": "log_workout",
                    "candidate_meal": None,
                    "candidate_workout": workout_info,
                    "analysis_text": response.content,
                    "confirmed": None,
                }
                state["final_response"] = response.content
                state["response"] = response.content
            else:
                state["requires_confirmation"] = False
                state["final_response"] = response.content
                state["response"] = state["final_response"]

        except Exception as e:
            print(f"[WorkoutAgent] 错误: {e}")
            state["response"] = "抱歉，健身指导服务暂时不可用，请稍后重试。"
            state["workout_result"] = None
            state["requires_confirmation"] = False

        # 更新对话历史
        if append_history:
            ctx_mgr = get_context_manager()
            ctx_mgr.append_messages(state, state["input_message"], state.get("response", ""))
        return state
