from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from agent.state import AgentState
from agent.router_agent import RouterAgent
from agent.food_agent import FoodAgent
from agent.workout_agent import WorkoutAgent
from agent.recipe_agent import RecipeAgent
from agent.memory_agent import get_memory_agent
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os


def get_postgres_checkpointer():
    """创建 PostgreSQL checkpointer

    从环境变量 DATABASE_URL 获取连接字符串，格式：
    postgresql://user:password@host:port/dbname

    Returns:
        PostgresSaver: PostgreSQL 检查点持久化器

    注意: 必须保持返回的 cm 全局引用，否则会被 GC 清理导致连接关闭。
    """
    global _postgres_cm
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    # Enter context to get actual saver instance; keep cm reference to avoid GC
    _postgres_cm = PostgresSaver.from_conn_string(db_url)
    saver = _postgres_cm.__enter__()
    # 初始化数据库表
    saver.setup()
    return saver


# 全局引用，防止 context manager 被 GC 清理导致连接关闭
_postgres_cm = None


def get_memory_checkpointer():
    """创建内存 checkpointer（仅用于测试或临时使用）"""
    from langgraph.checkpoint.memory import InMemorySaver
    return InMemorySaver()


# 默认使用 PostgreSQL checkpointer（生产环境）
# 注意：首次创建时需要数据库连接，连接失败会抛出异常
# 如果暂时没有数据库，回退到内存模式（状态重启后丢失，仅用于开发测试）
try:
    default_checkpointer = get_postgres_checkpointer()
except Exception as e:
    print(f"[Warning] PostgreSQL checkpointer init failed: {e}")
    print("[Warning] Falling back to InMemorySaver (state will be lost on restart)")
    default_checkpointer = get_memory_checkpointer()


def log_node(node_name: str):
    """打印节点执行日志"""
    print(f"\n{'='*50}")
    print(f"[Agent] 进入节点: {node_name}")
    print(f"{'='*50}\n")


# 意图到节点名的映射
INTENT_TO_NODE = {
    "food": "food_node",
    "workout": "workout_node",
    "recipe": "recipe_node",
    "stats_query": "stats_node",
    "confirm": "confirm_node",
    "profile_update": "profile_node",
    "general": "general_node"
}


def routing_func_multi(state: AgentState):
    """多意图路由 - 单意图直接处理，多意图路由到 multi_intent_node

    多意图时使用 multi_intent_node 内的线程池实现真正的并发
    """
    intents = state.get("intents", [state.get("intent", "general")])
    print(f"[Router] routing_func_multi - 多意图路由: {intents}")

    # 过滤出有效的意图节点（排除 confirm/profile/general/stats 这些需要特殊处理的）
    special_intents = {"confirm", "profile_update", "general", "stats_query"}
    intent_nodes = []
    has_special = False

    for intent in intents:
        if intent in special_intents:
            has_special = True
        if intent in INTENT_TO_NODE:
            node = INTENT_TO_NODE[intent]
            if node not in intent_nodes:
                intent_nodes.append(node)

    # 如果有特殊意图，交给对应节点处理（保持原有逻辑）
    if has_special:
        for intent in intents:
            if intent in special_intents and intent in INTENT_TO_NODE:
                print(f"[Router] routing_func_multi - 特殊意图 {intent}，路由到 {INTENT_TO_NODE[intent]}")
                return INTENT_TO_NODE[intent]

    if not intent_nodes:
        print(f"[Router] routing_func_multi - 无有效意图，使用 general_node")
        return "general_node"

    # 单节点
    if len(intent_nodes) == 1:
        print(f"[Router] routing_func_multi - 单意图，路由到 {intent_nodes[0]}")
        return intent_nodes[0]

    # 多节点 - 路由到 multi_intent_node，在那里用线程池并发执行
    print(f"[Router] routing_func_multi - 多意图，路由到 multi_intent_node")
    return "multi_intent_node"


def multi_intent_node(state: AgentState) -> AgentState:
    """多意图并发执行节点 - 使用线程池真正并发处理多个意图

    food 和 workout 同时执行，减少总等待时间
    """
    log_node("multi_intent_node (多意图并发执行)")

    intents = state.get("intents", [state.get("intent", "general")])
    print(f"[multi_intent_node] 待处理意图: {intents}")

    # 只处理 food 和 workout 的并发（这两个是耗时操作）
    has_food = "food" in intents
    has_workout = "workout" in intents

    if has_food and has_workout:
        # 真正并发：使用线程池同时执行两个 agent
        print(f"[multi_intent_node] 并发执行 food_node 和 workout_node")

        food_agent = FoodAgent()
        workout_agent = WorkoutAgent()

        # 创建独立的 state 副本给每个 agent
        food_state = dict(state)
        workout_state = dict(state)

        with ThreadPoolExecutor(max_workers=2) as executor:
            food_future = executor.submit(food_agent.run, food_state)
            workout_future = executor.submit(workout_agent.run, workout_state)

            # 等待两个都完成
            food_state = food_future.result()
            workout_state = workout_future.result()

        # 合并结果
        responses = []
        if food_state.get("food_result"):
            responses.append(food_state["food_result"])
            state["food_result"] = food_state["food_result"]
        if workout_state.get("workout_result"):
            responses.append(workout_state["workout_result"])
            state["workout_result"] = workout_state["workout_result"]

        # 合并 pending_stats（两个都可能设置）
        if food_state.get("pending_stats") and workout_state.get("pending_stats"):
            # 两个都有，合并
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

        # 合并响应
        if responses:
            state["response"] = "\n\n".join(responses)
        else:
            state["response"] = "已处理您的请求。"

    elif has_food:
        print(f"[multi_intent_node] 仅执行 food_node")
        food_agent = FoodAgent()
        food_agent.run(state)
        if state.get("food_result"):
            state["response"] = state["food_result"]

    elif has_workout:
        print(f"[multi_intent_node] 仅执行 workout_node")
        workout_agent = WorkoutAgent()
        workout_agent.run(state)
        if state.get("workout_result"):
            state["response"] = state["workout_result"]

    else:
        state["response"] = "已处理您的请求。"

    print(f"[multi_intent_node] 执行完成，response 长度: {len(state.get('response', '') if state.get('response') else 0)}")
    return state


def create_workflow(checkpointer=None):
    """创建langgraph工作流

    Args:
        checkpointer: 状态持久化检查点，默认使用 PostgreSQL（通过 DATABASE_URL 配置）
    """
    router = RouterAgent()
    food_agent = FoodAgent()
    workout_agent = WorkoutAgent()
    recipe_agent = RecipeAgent()

    builder = StateGraph(AgentState)
    builder.add_node("check_profile", router.check_profile)
    builder.add_node("init_daily_stats", init_daily_stats_node)
    builder.add_node("classify_intent", router.classify_intent)
    builder.add_node("food_node", food_agent.run)
    builder.add_node("workout_node", workout_agent.run)
    builder.add_node("recipe_node", recipe_agent.run)
    builder.add_node("confirm_node", router.handle_confirm)
    builder.add_node("profile_node", router.handle_profile_update)
    builder.add_node("general_node", router.handle_general)
    builder.add_node("stats_node", router.handle_stats_query)
    builder.add_node("multi_intent_node", multi_intent_node)

    builder.set_entry_point("check_profile")

    # 检查档案
    builder.add_conditional_edges(
        "check_profile",
        lambda state: "continue" if state.get("profile_complete", False) else "ask_profile",
        {
            "continue": "init_daily_stats",
            "ask_profile": "general_node"
        }
    )

    builder.add_edge("init_daily_stats", "classify_intent")

    # 意图分类后路由
    builder.add_conditional_edges(
        "classify_intent",
        routing_func_multi,
        {
            "food_node": END,
            "workout_node": END,
            "recipe_node": END,
            "stats_node": END,
            "confirm_node": END,
            "profile_node": END,
            "general_node": END,
            "multi_intent_node": END,
        }
    )

    # 编译时添加 checkpointer（如果传入）
    if checkpointer:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()


def init_daily_stats_node(state: AgentState) -> AgentState:
    """初始化每日统计节点"""
    log_node("init_daily_stats (初始化每日统计)")
    memory_agent = get_memory_agent()
    today = datetime.now().strftime("%Y-%m-%d")
    stats = memory_agent.load_daily_stats(today)
    state["daily_stats"] = stats
    return state
