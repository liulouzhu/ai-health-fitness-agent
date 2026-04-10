"""
多意图并发处理节点

使用 LangGraph Send API 实现真正的 fan-out / fan-in：
- fan-out 节点通过 Send 调度（由 routing_func 在 conditional_edges 中返回 Send）
- 各分支写自己专属的字段（food_branch_result / workout_branch_result / stats_branch_result）
- 各分支的 pending 上下文也写自己专属字段（food_pending_conf / workout_pending_conf）
- multi_join_node 读取分支专属字段，合成统一确认上下文

节点职责：
- food_workout_fanout: food + workout 并行 → join
- food_stats_fanout: food + stats_query 并行 → join
- workout_stats_fanout: workout + stats_query 并行 → join
- multi_join_node: 读取各分支结果 + pending_conf → 合成 final_response + 统一 pending_confirmation
"""

from langgraph.types import Send, Command
from agent.state import AgentState
from agent.food_agent import FoodAgent
from agent.workout_agent import WorkoutAgent
from agent.memory import get_memory_agent

# ============ 共享工具 ============

_CLEARED_FIELDS = (
    "food_branch_result", "workout_branch_result", "stats_branch_result",
    "food_pending_conf", "workout_pending_conf",
    "branch_results", "food_result", "workout_result", "stats_result",
    "pending_confirmation", "requires_confirmation", "pending_stats",
)


def _clear_state(state: dict) -> dict:
    """清除分支残留字段，避免并发写入冲突"""
    for key in _CLEARED_FIELDS:
        state.pop(key, None)
    return state


# ============ Fan-out 构建器 ============

def _make_fanout(fanout_name: str, branch_mapping: list):
    """
    通用 fan-out 工厂。

    branch_mapping: list of (branch_name, intent_filter) tuples
        e.g. [("food_branch", ["food", "food_report"]), ("workout_branch", ["workout", "workout_report"])]
    """
    def fanout(state: AgentState) -> Command:
        state["route_decision"] = fanout_name
        intents = state.get("intents", [state.get("intent", "general")])
        print(f"[Fanout] {fanout_name} - intents={intents}")

        sends = []
        for branch_name, intent_filter in branch_mapping:
            filtered = [i for i in intents if i in intent_filter]
            if filtered:
                branch_state = _clear_state(dict(state))
                branch_state["intent"] = filtered[0]
                sends.append(Send(branch_name, branch_state))
        return Command(goto=sends)

    return fanout


# ============ 分支节点 ============

def food_branch(state: AgentState) -> AgentState:
    """Food 分支节点（由 Send 并发调度）"""
    print(f"[Branch] food_branch 开始")
    food_agent = FoodAgent()
    state = food_agent.run(state)
    return {
        "food_branch_result": state.get("food_result", ""),
        "food_pending_conf": state.get("pending_confirmation"),
    }


def workout_branch(state: AgentState) -> AgentState:
    """Workout 分支节点（由 Send 并发调度）"""
    print(f"[Branch] workout_branch 开始")
    workout_agent = WorkoutAgent()
    state = workout_agent.run(state)
    return {
        "workout_branch_result": state.get("workout_result", ""),
        "workout_pending_conf": state.get("pending_confirmation"),
    }


def stats_branch(state: AgentState) -> AgentState:
    """Stats 分支节点"""
    print(f"[Branch] stats_branch 开始")
    memory_agent = get_memory_agent()
    summary = memory_agent.get_daily_summary()
    return {
        "stats_branch_result": summary,
        "stats_result": summary,
    }


# ============ 具体 fan-out 实例 ============

food_workout_fanout = _make_fanout(
    "food_workout_fanout",
    [("food_branch", ["food", "food_report"]), ("workout_branch", ["workout", "workout_report"])],
)

food_stats_fanout = _make_fanout(
    "food_stats_fanout",
    [("food_branch", ["food", "food_report"]), ("stats_branch", ["stats_query"])],
)

workout_stats_fanout = _make_fanout(
    "workout_stats_fanout",
    [("workout_branch", ["workout", "workout_report"]), ("stats_branch", ["stats_query"])],
)


# ============ 汇合节点 ============

def multi_join_node(state: AgentState) -> dict:
    """
    多意图 fan-out 后的汇合节点。

    从分支专属字段读取各分支结果：
    - food_branch_result / workout_branch_result / stats_branch_result
    - food_pending_conf / workout_pending_conf

    生成最终响应和统一的 pending_confirmation（用于 confirm 流程）。

    返回增量字段而非完整 state，防止 checkpoint 过大。
    """
    food_result = state.get("food_branch_result", "")
    workout_result = state.get("workout_branch_result", "")
    stats_result = state.get("stats_branch_result", "")
    food_conf = state.get("food_pending_conf") or {}
    workout_conf = state.get("workout_pending_conf") or {}

    print(f"[Join] multi_join_node - food_result={bool(food_result)}, "
          f"workout_result={bool(workout_result)}, stats_result={bool(stats_result)}")

    # 合并各分支响应
    sections = []
    if food_result:
        sections.append(f"**食物记录**\n{food_result}")
    if workout_result:
        sections.append(f"**运动记录**\n{workout_result}")
    if stats_result:
        sections.append(f"**今日统计**\n{stats_result}")

    final_response = "\n\n".join(sections) if sections else "已处理您的请求。"

    # 确定 pending_confirmation（analysis_text 只保留摘要，不保留全文）
    unified_conf = None
    has_food = bool(food_conf.get("action"))
    has_workout = bool(workout_conf.get("action"))

    if has_food and has_workout:
        unified_conf = {
            "action": "log_both",
            "candidate_meal": food_conf.get("candidate_meal"),
            "candidate_workout": workout_conf.get("candidate_workout"),
            "analysis_text": "（含食物和运动记录，请确认）",
            "confirmed": None,
        }
    elif has_food:
        unified_conf = {
            "action": food_conf.get("action"),
            "candidate_meal": food_conf.get("candidate_meal"),
            "candidate_workout": None,
            "analysis_text": "（请确认是否计入）",
            "confirmed": None,
        }
    elif has_workout:
        unified_conf = {
            "action": workout_conf.get("action"),
            "candidate_meal": None,
            "candidate_workout": workout_conf.get("candidate_workout"),
            "analysis_text": "（请确认是否计入）",
            "confirmed": None,
        }

    if unified_conf:
        final_response += "\n\n---\n是否将上述记录计入今日统计？（是/否）"

    # 返回增量字段，不包含 state 完整副本
    return {
        "route_decision": "multi_join_node",
        "final_response": final_response,
        "response": final_response,
        "pending_confirmation": unified_conf,
        "requires_confirmation": bool(unified_conf),
        # 清理分支结果字段
        "food_branch_result": None,
        "workout_branch_result": None,
        "stats_branch_result": None,
        "food_pending_conf": None,
        "workout_pending_conf": None,
        "food_result": None,
        "workout_result": None,
        "stats_result": None,
        "recipe_result": None,
    }
