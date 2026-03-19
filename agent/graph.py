from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.router_agent import RouterAgent
from agent.food_agent import FoodAgent
from agent.workout_agent import WorkoutAgent


def create_workflow():
    """创建langgraph工作流"""
    # 初始化agents
    router = RouterAgent()
    food_agent = FoodAgent()
    workout_agent = WorkoutAgent()

    # 构建图
    builder = StateGraph(AgentState)
    builder.add_node("check_profile", router.check_profile)
    builder.add_node("classify_intent", router.classify_intent)
    builder.add_node("food_node", food_agent.run)
    builder.add_node("workout_node", workout_agent.run)
    builder.add_node("profile_node", router.handle_profile_update)
    builder.add_node("general_node", router.handle_general)

    # 设置入口点
    builder.set_entry_point("check_profile")

    # 条件边：检查档案后分流
    builder.add_conditional_edges(
        "check_profile",
        lambda state: "continue" if state.get("profile_complete", False) else "ask_profile",
        {
            "continue": "classify_intent",
            "ask_profile": "general_node"
        }
    )

    # 意图分类后的条件边
    builder.add_conditional_edges(
        "classify_intent",
        router.routing_func,
        {
            "food": "food_node",
            "workout": "workout_node",
            "profile_update": "profile_node",
            "general": "general_node"
        }
    )

    # 普通边 - 结束后结束
    builder.add_edge("food_node", END)
    builder.add_edge("workout_node", END)
    builder.add_edge("profile_node", END)
    builder.add_edge("general_node", END)

    # 编译图
    return builder.compile()
