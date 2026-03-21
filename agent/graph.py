from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from agent.state import AgentState
from agent.router_agent import RouterAgent
from agent.food_agent import FoodAgent
from agent.workout_agent import WorkoutAgent
from agent.recipe_agent import RecipeAgent
from agent.memory_agent import get_memory_agent
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
