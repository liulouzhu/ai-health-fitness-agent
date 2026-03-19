from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from agent.state import AgentState
from agent.router_agent import RouterAgent
from agent.food_agent import FoodAgent
from agent.workout_agent import WorkoutAgent
from agent.recipe_agent import RecipeAgent
from agent.memory_agent import get_memory_agent
from datetime import datetime


# 默认 checkpointer
default_checkpointer = InMemorySaver()


def log_node(node_name: str):
    """打印节点执行日志"""
    print(f"\n{'='*50}")
    print(f"[Agent] 进入节点: {node_name}")
    print(f"{'='*50}\n")


def create_workflow(checkpointer=None):
    """创建langgraph工作流

    Args:
        checkpointer: 状态持久化检查点，默认使用 InMemorySaver
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
        router.routing_func,
        {
            "food": "food_node",
            "workout": "workout_node",
            "recipe": "recipe_node",
            "stats_query": "stats_node",
            "confirm": "confirm_node",
            "profile_update": "profile_node",
            "general": "general_node"
        }
    )

    # food/workout 节点后直接结束，等待用户确认
    builder.add_edge("food_node", END)
    builder.add_edge("workout_node", END)

    # recipe 节点直接结束（不需要确认）
    builder.add_edge("recipe_node", END)

    # 确认节点单独处理
    builder.add_edge("confirm_node", END)
    builder.add_edge("profile_node", END)
    builder.add_edge("general_node", END)
    builder.add_edge("stats_node", END)

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
