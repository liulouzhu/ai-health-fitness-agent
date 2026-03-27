"""多意图并发处理节点

每个意图组合对应独立节点，通过 LangGraph Send API 实现原生并行：
- food_workout_node: 并发执行 food + workout
- food_stats_node: 并发执行 food + stats_query
- workout_stats_node: 并发执行 workout + stats_query

并行节点只写 xxx_result 字段（不写 response），multi_join_node 负责合并生成最终响应。
"""

from agent.state import AgentState
from agent.food_agent import FoodAgent
from agent.workout_agent import WorkoutAgent
from agent.memory_agent import get_memory_agent


# ============ 辅助函数 ============

def _save_pending_if_exists(state: AgentState) -> None:
    """如有 pending_stats 则持久化"""
    pending = state.get("pending_stats")
    if pending:
        get_memory_agent().save_pending_stats(pending)


# ============ 并行执行节点（通过 Send 调用）============

def food_workout_node(state: AgentState) -> AgentState:
    """并发执行 food + workout，通过 Send fan-out 由 LangGraph 调度"""
    intents = state.get("intents", [state.get("intent", "general")])

    food_agent = FoodAgent()
    food_state = dict(state)
    if "food_report" in intents:
        food_state["intent"] = "food_report"
    food_agent.run(food_state)

    workout_agent = WorkoutAgent()
    workout_state = dict(state)
    if "workout_report" in intents:
        workout_state["intent"] = "workout_report"
    workout_agent.run(workout_state)

    state["food_result"] = food_state.get("food_result")
    state["workout_result"] = workout_state.get("workout_result")
    state["pending_stats"] = workout_state.get("pending_stats") or food_state.get("pending_stats")
    state["messages"] = food_state.get("messages", []) + workout_state.get("messages", [])

    _save_pending_if_exists(state)
    return state


def food_stats_node(state: AgentState) -> AgentState:
    """并发执行 food + stats_query，通过 Send fan-out 由 LangGraph 调度"""
    intents = state.get("intents", [state.get("intent", "general")])

    food_agent = FoodAgent()
    food_state = dict(state)
    if "food_report" in intents:
        food_state["intent"] = "food_report"
    food_agent.run(food_state)

    stats_state = dict(state)
    stats_state["response"] = get_memory_agent().get_daily_summary()

    state["food_result"] = food_state.get("food_result")
    state["stats_result"] = stats_state.get("response")
    state["pending_stats"] = food_state.get("pending_stats")
    state["messages"] = food_state.get("messages", []) + stats_state.get("messages", [])

    _save_pending_if_exists(state)
    return state


def workout_stats_node(state: AgentState) -> AgentState:
    """并发执行 workout + stats_query，通过 Send fan-out 由 LangGraph 调度"""
    intents = state.get("intents", [state.get("intent", "general")])

    workout_agent = WorkoutAgent()
    workout_state = dict(state)
    if "workout_report" in intents:
        workout_state["intent"] = "workout_report"
    workout_agent.run(workout_state)

    stats_state = dict(state)
    stats_state["response"] = get_memory_agent().get_daily_summary()

    state["workout_result"] = workout_state.get("workout_result")
    state["stats_result"] = stats_state.get("response")
    state["pending_stats"] = workout_state.get("pending_stats")
    state["messages"] = workout_state.get("messages", []) + stats_state.get("messages", [])

    _save_pending_if_exists(state)
    return state


# ============ 汇合节点 ============

def _merge_responses_structured(results: dict) -> str:
    """结构化合并多意图响应"""
    sections = []

    if results.get("food"):
        sections.append("**食物记录**\n" + results["food"])

    if results.get("workout"):
        sections.append("**运动记录**\n" + results["workout"])

    if results.get("stats"):
        sections.append("**今日统计**\n" + results["stats"])

    return "\n\n".join(sections) if sections else "已处理您的请求。"


def multi_join_node(state: AgentState) -> AgentState:
    """多意图并行执行后的汇合节点"""
    results = {}
    if state.get("food_result"):
        results["food"] = state["food_result"]
    if state.get("workout_result"):
        results["workout"] = state["workout_result"]
    if state.get("stats_result"):
        results["stats"] = state["stats_result"]

    state["response"] = _merge_responses_structured(results)

    # 清理各 result 字段，避免污染下游
    state.pop("food_result", None)
    state.pop("workout_result", None)
    state.pop("stats_result", None)

    return state
