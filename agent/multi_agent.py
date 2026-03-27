"""多意图并发处理节点

包含 multi_intent_node 及相关的并发处理函数
"""

from concurrent.futures import ThreadPoolExecutor
from agent.state import AgentState
from agent.router_agent import RouterAgent
from agent.food_agent import FoodAgent
from agent.workout_agent import WorkoutAgent
from agent.memory_agent import get_memory_agent


def log_node(node_name: str):
    """打印节点执行日志"""
    print(f"\n{'='*50}")
    print(f"[Agent] 进入节点: {node_name}")
    print(f"{'='*50}\n")


def _merge_responses_structured(results: dict) -> str:
    """结构化合并多意图响应"""
    sections = []

    if results.get("food"):
        sections.append("🍽️ **食物记录**\n" + results["food"])

    if results.get("workout"):
        sections.append("🏃 **运动记录**\n" + results["workout"])

    if results.get("stats"):
        sections.append("📊 **今日统计**\n" + results["stats"])

    return "\n\n".join(sections) if sections else "已处理您的请求。"


def _handle_food_workout_concurrent(state: AgentState, intents: list):
    """并发处理 food + workout"""
    print(f"[multi_intent_node] 并发执行 food_node 和 workout_node")

    food_agent = FoodAgent()
    workout_agent = WorkoutAgent()

    food_state = dict(state)
    workout_state = dict(state)
    if "food_report" in intents:
        food_state["intent"] = "food_report"
    if "workout_report" in intents:
        workout_state["intent"] = "workout_report"

    with ThreadPoolExecutor(max_workers=2) as executor:
        food_future = executor.submit(food_agent.run, food_state)
        workout_future = executor.submit(workout_agent.run, workout_state)
        food_state = food_future.result()
        workout_state = workout_future.result()

    results = {}
    if food_state.get("response"):
        results["food"] = food_state["response"]
        state["food_result"] = food_state.get("food_result")
    if workout_state.get("response"):
        results["workout"] = workout_state["response"]
        state["workout_result"] = workout_state.get("workout_result")

    # 合并 pending_stats
    if food_state.get("pending_stats") and workout_state.get("pending_stats"):
        state["pending_stats"] = {
            "type": "multi",
            "food": food_state["pending_stats"].get("data"),
            "workout": workout_state["pending_stats"].get("data"),
            "responses": [food_state["pending_stats"].get("response"), workout_state["pending_stats"].get("response")]
        }
    elif food_state.get("pending_stats"):
        state["pending_stats"] = food_state["pending_stats"]
    elif workout_state.get("pending_stats"):
        state["pending_stats"] = workout_state["pending_stats"]

    if state.get("pending_stats"):
        get_memory_agent().save_pending_stats(state["pending_stats"])

    state["response"] = _merge_responses_structured(results)


def _handle_food_stats_concurrent(state: AgentState, intents: list):
    """并发处理 food + stats_query"""
    print(f"[multi_intent_node] 并发执行 food_node 和 stats_query")

    router_agent = RouterAgent()
    food_agent = FoodAgent()

    food_state = dict(state)
    stats_state = dict(state)
    if "food_report" in intents:
        food_state["intent"] = "food_report"

    with ThreadPoolExecutor(max_workers=2) as executor:
        food_future = executor.submit(food_agent.run, food_state)
        stats_future = executor.submit(router_agent.handle_stats_query, stats_state)
        food_result_state = food_future.result()
        stats_result_state = stats_future.result()

    results = {}
    if food_result_state.get("response"):
        results["food"] = food_result_state["response"]
    if stats_result_state.get("response"):
        results["stats"] = stats_result_state["response"]

    state["pending_stats"] = food_result_state.get("pending_stats")
    state["response"] = _merge_responses_structured(results)
    state["messages"] = food_result_state.get("messages", state.get("messages", []))


def _handle_workout_stats_concurrent(state: AgentState, intents: list):
    """并发处理 workout + stats_query"""
    print(f"[multi_intent_node] 并发执行 workout_node 和 stats_query")

    router_agent = RouterAgent()
    workout_agent = WorkoutAgent()

    workout_state = dict(state)
    stats_state = dict(state)
    if "workout_report" in intents:
        workout_state["intent"] = "workout_report"

    with ThreadPoolExecutor(max_workers=2) as executor:
        workout_future = executor.submit(workout_agent.run, workout_state)
        stats_future = executor.submit(router_agent.handle_stats_query, stats_state)
        workout_result_state = workout_future.result()
        stats_result_state = stats_future.result()

    results = {}
    if workout_result_state.get("response"):
        results["workout"] = workout_result_state["response"]
    if stats_result_state.get("response"):
        results["stats"] = stats_result_state["response"]

    state["pending_stats"] = workout_result_state.get("pending_stats")
    state["response"] = _merge_responses_structured(results)
    state["messages"] = workout_result_state.get("messages", state.get("messages", []))


def _handle_single_intent(state: AgentState, intent: str, intents: list):
    """处理单一意图（food 或 workout）"""
    print(f"[multi_intent_node] 仅执行 {intent}_node")
    if intent == "food":
        FoodAgent().run(state)
    else:
        WorkoutAgent().run(state)


def multi_intent_node(state: AgentState) -> AgentState:
    """多意图并发执行节点 - 使用线程池真正并发处理多个意图"""
    log_node("multi_intent_node (多意图并发执行)")

    intents = state.get("intents", [state.get("intent", "general")])
    print(f"[multi_intent_node] 待处理意图: {intents}")

    has_food = "food" in intents or "food_report" in intents
    has_workout = "workout" in intents or "workout_report" in intents
    has_stats = "stats_query" in intents

    if has_food and has_workout:
        _handle_food_workout_concurrent(state, intents)
    elif has_food and has_stats:
        _handle_food_stats_concurrent(state, intents)
    elif has_workout and has_stats:
        _handle_workout_stats_concurrent(state, intents)
    elif has_food:
        _handle_single_intent(state, "food", intents)
    elif has_workout:
        _handle_single_intent(state, "workout", intents)
    elif has_stats:
        router_agent = RouterAgent()
        router_agent.handle_stats_query(state)
        if not state.get("response"):
            state["response"] = "已处理您的请求。"
    else:
        state["response"] = "已处理您的请求。"

    print(f"[multi_intent_node] 执行完成，response 长度: {len(state.get('response', '') if state.get('response') else 0)}")
    return state
