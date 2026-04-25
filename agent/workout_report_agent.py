"""
运动记录 Agent

职责：
- 从用户输入中提取运动数据（类型、时长、消耗）
- 生成运动记录确认信息
- 设置 pending_confirmation / requires_confirmation
- 由 graph 的 confirm_node → commit_node 完成实际写入

只处理 workout_report 意图，不回答运动咨询问题。
"""

from pydantic import BaseModel
from agent.llm import get_llm
from agent.state import AgentState
from agent.context_manager import get_context_manager


class WorkoutInfo(BaseModel):
    """运动信息结构"""
    type: str
    duration: float
    calories: float


class WorkoutReportAgent:
    def __init__(self):
        self.llm = get_llm()
        self.llm_with_tools = self.llm.bind_tools([WorkoutInfo])

    def _extract_workout_info(self, user_input: str) -> dict:
        """从用户输入中提取运动信息（双重策略：tool calling + 文本解析 fallback）"""
        import re

        # 策略1：tool calling
        try:
            response = self.llm_with_tools.invoke([
                {"role": "user", "content": f"从以下文本中提取运动信息（类型、时长分钟、消耗千卡）：{user_input}"}
            ])
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                args = tool_call.get("args", {}) or tool_call.get("arguments", {})
                if isinstance(args, str):
                    import json
                    args = json.loads(args)
                result = {
                    "type": args.get("type", "未知运动"),
                    "duration": float(args.get("duration", 0) or 0),
                    "calories": float(args.get("calories", 0) or 0)
                }
                if result["duration"] > 0 or result["calories"] > 0:
                    return result
            print(f"[WorkoutReportAgent] tool calling 未返回有效数据，尝试文本解析")
        except Exception as e:
            print(f"[WorkoutReportAgent] tool calling 失败: {e}，尝试文本解析")

        # 策略2：正则提取距离/时长，用经验公式估算
        distance_km = 0
        duration_min = 0
        workout_type = "运动"

        # 提取距离
        km_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:公里|km|千米)", user_input)
        if km_match:
            distance_km = float(km_match.group(1))
            workout_type = "跑步"

        # 提取时长
        min_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:分钟|min)", user_input)
        if min_match:
            duration_min = float(min_match.group(1))

        # 用距离估算（跑步约60kcal/km）
        if distance_km > 0 and duration_min == 0:
            duration_min = distance_km * 6  # 约6min/km
        if distance_km > 0:
            calories = round(distance_km * 60)
        elif duration_min > 0:
            calories = round(duration_min * 8)  # 约8kcal/min
        else:
            calories = 0
            duration_min = 0

        if calories > 0 or duration_min > 0:
            print(f"[WorkoutReportAgent] 文本解析成功: type={workout_type}, duration={duration_min}, calories={calories}")
            return {"type": workout_type, "duration": duration_min, "calories": calories}

        return {"type": "未知运动", "duration": 0, "calories": 0}

    def run(
        self,
        state: AgentState,
        extra_sections: dict = None,
        branch_input: str = None,
        append_history: bool = False,
    ) -> AgentState:
        """处理运动记录

        流程：
        1. 提取运动数据
        2. LLM 生成记录确认信息
        3. 设置 pending_confirmation（auto-confirm）
        """
        print(f"[WorkoutReportAgent] run - 处理运动记录")

        query = branch_input or state["input_message"]

        try:
            # === Step 1: 提取运动数据 ===
            workout_info = self._extract_workout_info(query)
            print(f"[WorkoutReportAgent] 提取结果: {workout_info}")

            # === Step 2: LLM 生成确认信息 ===
            ctx_mgr = get_context_manager()
            if extra_sections is not None:
                extra_sections = dict(extra_sections)
            else:
                preferences = ctx_mgr.get_preferences_str() or "（暂无偏好记录）"
                extra_sections = {
                    "用户偏好": preferences,
                }

            messages = ctx_mgr.build_prompt_messages(
                "workout_report",
                state,
                extra_sections=extra_sections,
                user_input=query,
            )
            response = self.llm.invoke(messages)
            state["workout_result"] = response.content

            # === Step 3: 设置确认态（auto-confirm） ===
            if workout_info.get("duration", 0) > 0 or workout_info.get("calories", 0) > 0:
                state["requires_confirmation"] = True
                state["pending_confirmation"] = {
                    "action": "log_workout",
                    "candidate_meal": None,
                    "candidate_workout": workout_info,
                    "analysis_text": response.content,
                    "confirmed": True,
                }
            else:
                state["requires_confirmation"] = False
                state["final_response"] = response.content
                state["response"] = response.content

        except Exception as e:
            print(f"[WorkoutReportAgent] 错误: {e}")
            state["response"] = "抱歉，运动记录处理失败，请稍后重试。"
            state["workout_result"] = None
            state["requires_confirmation"] = False

        return state
