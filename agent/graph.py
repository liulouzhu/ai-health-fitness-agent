from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Send, Command
from agent.state import AgentState, PendingConfirmation
from agent.router_agent import RouterAgent, CONFIRM_WORDS, DENY_WORDS
from agent.food_agent import FoodAgent
from agent.workout_agent import WorkoutAgent
from agent.recipe_agent import RecipeAgent
from agent.multi_agent import (
    food_workout_fanout, food_stats_fanout, workout_stats_fanout,
    food_branch, workout_branch, stats_branch,
    multi_join_node
)
from agent.memory import get_memory_agent
from datetime import datetime
import os


def log_node(node_name: str):
    """打印节点执行日志"""
    print(f"\n{'='*50}")
    print(f"[Graph] >>> 节点: {node_name}")
    print(f"{'='*50}\n")


# ============ Checkpointer ============

_postgres_cm = None


def get_postgres_checkpointer():
    """创建 PostgreSQL checkpointer（从 DATABASE_URL 获取连接字符串）

    postgresql://user:password@host:port/dbname
    首次创建时需要数据库连接，连接失败会回退到内存模式。
    """
    global _postgres_cm
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    _postgres_cm = PostgresSaver.from_conn_string(db_url)
    saver = _postgres_cm.__enter__()
    saver.setup()
    return saver


def get_memory_checkpointer():
    """创建内存 checkpointer（仅用于测试或临时使用）"""
    from langgraph.checkpoint.memory import InMemorySaver
    return InMemorySaver()


# 默认 checkpointer（生产用 PostgreSQL，失败则内存）
try:
    default_checkpointer = get_postgres_checkpointer()
except Exception as e:
    print(f"[Warning] PostgreSQL checkpointer init failed: {e}")
    print("[Warning] Falling back to InMemorySaver (state will be lost on restart)")
    default_checkpointer = get_memory_checkpointer()


# ============ 路由函数 ============

# 意图 → 节点名 映射（单意图）
INTENT_TO_NODE = {
    "food": "food_generate",
    "food_report": "food_generate",
    "workout": "workout_generate",
    "workout_report": "workout_generate",
    "recipe": "recipe_node",
    "stats_query": "stats_node",
    "confirm": "confirm_node",
    "profile_update": "profile_node",
    "general": "general_node",
}


def routing_func(state: AgentState):
    """主路由函数 - LangGraph 条件路由

    多意图时返回 [Send(...)] 实现原生 fan-out。
    单意图时返回节点名字符串。

    特殊意图（confirm / profile_update / general）优先处理，
    如果和其他意图混合，整体路由到 general_node。
    """
    intents = state.get("intents", [state.get("intent", "general")])
    print(f"[Router] routing_func - intents: {intents}")

    # 清理空意图
    intents = [i for i in intents if i]
    if not intents:
        print("[Router] 无有效意图 → general_node")
        return "general_node"

    # ---- 意图等价类标准化 ----
    # food_report / workout_report 应等价于 food / workout，支持 fan-out
    _intent_aliases = {
        "food_report": "food",
        "workout_report": "workout",
    }
    normalized_intents = []
    for i in intents:
        normalized_intents.append(_intent_aliases.get(i, i))
    intents = normalized_intents
    print(f"[Router] 标准化意图: {intents}")

    # ---- 特殊意图处理 ----
    special_intents = {"confirm", "profile_update"}
    regular_intents = set(i for i in intents if i not in special_intents)

    # 纯特殊意图：单独处理
    if not regular_intents:
        intent = intents[0]
        if intent == "confirm":
            # pending_confirmation 非空且 confirmed=None → 在等待确认回复
            # → confirm_recovery 处理用户回复
            # 否则 → confirm_node 显示/重新显示确认提示
            pending_conf = state.get("pending_confirmation") or {}
            if pending_conf and pending_conf.get("confirmed") is None:
                return "confirm_recovery"
            return "confirm_node"
        elif intent == "profile_update":
            return "profile_node"

    # 混合意图（special + regular）：降级为 general_node
    special_user_intents = set(intents) & special_intents
    if special_user_intents:
        print(f"[Router] 混合意图 {intents}，降级到 general_node")
        return "general_node"

    # ---- 常规意图路由 ----
    # 单意图
    if len(intents) == 1:
        intent = intents[0]
        node = INTENT_TO_NODE.get(intent)
        if node:
            print(f"[Router] 单意图 {intent} → {node}")
            return node
        print(f"[Router] 单意图 {intent} 无对应节点 → general_node")
        return "general_node"

    # ---- 多意图 fan-out ----
    # 返回字符串节点名（由条件边路由），由 fan-out 节点内部发射 Send
    regular_set = regular_intents
    print(f"[Router] 多意图 {intents} → fan-out")

    if regular_set == {"food", "workout"}:
        return "food_workout_fanout"
    if regular_set == {"food", "stats_query"}:
        return "food_stats_fanout"
    if regular_set == {"workout", "stats_query"}:
        return "workout_stats_fanout"
    if regular_set == {"food", "workout", "stats_query"}:
        return "food_workout_fanout"

    # 其他组合：降级
    print(f"[Router] 意图组合 {regular_set} 降级到 general_node")
    return "general_node"


def profile_check_route(state: AgentState) -> str:
    """profile 检查后的路由（直接从磁盘读取，避免 checkpointer 缓存不一致）"""
    memory_agent = get_memory_agent()
    if memory_agent.is_profile_complete():
        return "init_daily_stats"
    return "general_node"


# ============ 节点定义 ============

def init_daily_stats_node(state: AgentState) -> AgentState:
    """初始化每日统计节点"""
    log_node("init_daily_stats")
    state["route_decision"] = "init_daily_stats"
    memory_agent = get_memory_agent()
    today = datetime.now().strftime("%Y-%m-%d")
    stats = memory_agent.load_daily_stats(today)
    state["daily_stats"] = stats
    return state


def food_generate_node(state: AgentState) -> AgentState:
    """食物分析 + 生成候选结果节点

    负责：
    1. 检索/分析食物
    2. 提取营养数据
    3. 生成候选记录
    4. 设置 requires_confirmation / pending_action
    5. 写入 final_response（待确认提示文本）
    """
    log_node("food_generate")
    state["route_decision"] = "food_generate"
    food_agent = FoodAgent()
    state = food_agent.run(state)
    return state


def workout_generate_node(state: AgentState) -> AgentState:
    """运动指导 + 生成候选结果节点"""
    log_node("workout_generate")
    state["route_decision"] = "workout_generate"
    workout_agent = WorkoutAgent()
    state = workout_agent.run(state)
    return state


def stats_node(state: AgentState) -> AgentState:
    """统计查询节点"""
    log_node("stats_node")
    state["route_decision"] = "stats_node"
    memory_agent = get_memory_agent()
    summary = memory_agent.get_daily_summary()
    state["final_response"] = summary
    state["response"] = summary
    return state


def recipe_node(state: AgentState) -> AgentState:
    """食谱推荐节点"""
    log_node("recipe_node")
    state["route_decision"] = "recipe_node"
    recipe_agent = RecipeAgent()
    state = recipe_agent.run(state)
    state["final_response"] = state.get("response")
    return state


def profile_node(state: AgentState) -> AgentState:
    """档案更新节点"""
    log_node("profile_node")
    state["route_decision"] = "profile_node"
    router = RouterAgent()
    state = router.handle_profile_update(state)
    state["final_response"] = state.get("response")
    return state


def general_node(state: AgentState) -> AgentState:
    """通用对话节点"""
    log_node("general_node")
    state["route_decision"] = "general_node"
    router = RouterAgent()
    state = router.handle_general(state)
    state["final_response"] = state.get("response")
    return state


def confirm_node(state: AgentState) -> AgentState:
    """确认节点 - 展示待确认内容并等待用户回复

    首次进入：设置 pending_confirmation，显示确认提示
    routing_func 根据 pending_confirmation.confirmed 是否为 None 判断是否在等待回复
    """
    log_node("confirm_node")
    state["route_decision"] = "confirm_node"

    pending_action = state.get("pending_action")
    candidate_meal = state.get("candidate_meal")
    candidate_workout = state.get("candidate_workout")

    # 设置 pending_confirmation（confirmed=None 表示"等待回复"）
    # routing_func 根据 pending_confirmation 是否存在来判断：
    # - 有数据且 confirmed=None → 用户在回复确认提示 → routing_func 会路由到 confirm_recovery
    # - 无数据 → 新确认周期开始 → 显示确认提示
    if pending_action == "log_meal" and candidate_meal:
        analysis = state.get("food_result", state.get("response", ""))
        state["pending_confirmation"] = {
            "action": "log_meal",
            "candidate_meal": candidate_meal,
            "candidate_workout": None,
            "analysis_text": analysis,
            "confirmed": None,
        }
        state["final_response"] = (
            f"{analysis}\n\n---\n"
            f"是否将上述食物计入今日热量统计？（是/否）"
        )
    elif pending_action == "log_workout" and candidate_workout:
        analysis = state.get("workout_result", state.get("response", ""))
        state["pending_confirmation"] = {
            "action": "log_workout",
            "candidate_meal": None,
            "candidate_workout": candidate_workout,
            "analysis_text": analysis,
            "confirmed": None,
        }
        state["final_response"] = (
            f"{analysis}\n\n---\n"
            f"是否将上述运动计入今日消耗统计？（是/否）"
        )
    elif pending_action == "log_both":
        food_analysis = state.get("food_result", "")
        workout_analysis = state.get("workout_result", "")
        state["pending_confirmation"] = {
            "action": "log_both",
            "candidate_meal": candidate_meal,
            "candidate_workout": candidate_workout,
            "analysis_text": f"{food_analysis}\n\n{workout_analysis}",
            "confirmed": None,
        }
        state["final_response"] = (
            f"**食物记录**\n{food_analysis}\n\n---\n"
            f"**运动记录**\n{workout_analysis}\n\n---\n"
            f"是否将上述记录计入今日统计？（是/否）"
        )
    else:
        # 没有待确认内容
        state["final_response"] = "没有待确认的记录。"
        state["requires_confirmation"] = False
        state["pending_action"] = None

    state["response"] = state["final_response"]
    return state


def commit_node(state: AgentState) -> AgentState:
    """提交节点 - 将确认后的记录写入 daily_stats"""
    log_node("commit_node")
    state["route_decision"] = "commit_node"
    memory_agent = get_memory_agent()
    pending_conf = state.get("pending_confirmation", {})
    confirmed = pending_conf.get("confirmed")

    if confirmed is True:
        action = pending_conf.get("action")
        if action in ("log_meal", "log_both"):
            meal = pending_conf.get("candidate_meal")
            if meal:
                memory_agent.update_daily_stats("meal", meal)
        if action in ("log_workout", "log_both"):
            workout = pending_conf.get("candidate_workout")
            if workout:
                memory_agent.update_daily_stats("workout", workout)

        summary = memory_agent.get_daily_summary()
        state["final_response"] = f"已计入统计。\n\n{summary}"
        state["response"] = state["final_response"]
        print("[Commit] 已写入 daily_stats")
    elif confirmed is False:
        state["final_response"] = "好的，已取消。"
        state["response"] = state["final_response"]
        print("[Commit] 用户取消")
    else:
        # 状态异常：不应该直接到达这里
        state["final_response"] = "确认状态异常。"
        state["response"] = state["final_response"]

    # 清理确认状态
    state["requires_confirmation"] = False
    state["pending_action"] = None
    state["candidate_meal"] = None
    state["candidate_workout"] = None
    state["pending_confirmation"] = {}
    state["pending_stats"] = None

    # 清理旧版 pending_stats
    try:
        memory_agent.clear_pending_stats()
    except Exception:
        pass

    return state


def confirm_recovery_node(state: AgentState) -> AgentState:
    """确认恢复节点 - 处理用户对 confirm_node 问题的回复

    当用户回复"是/好/确认"或"取消/不算"时，这个节点读取回复内容，
    设置 pending_confirmation.confirmed 标志，然后流向 commit_node。
    """
    log_node("confirm_recovery")
    state["route_decision"] = "confirm_recovery"

    # 从 state 读取用户原始输入
    user_input = state.get("input_message", "").strip().lower()

    is_deny = any(word == user_input for word in DENY_WORDS)
    is_yes = not is_deny and any(word == user_input for word in CONFIRM_WORDS)

    # 获取之前的 pending_confirmation
    pending_conf = dict(state.get("pending_confirmation") or {})

    if is_yes:
        pending_conf["confirmed"] = True
        state["pending_confirmation"] = pending_conf
        state["requires_confirmation"] = False
        print("[ConfirmRecovery] 用户确认")
    elif is_deny:
        pending_conf["confirmed"] = False
        state["pending_confirmation"] = pending_conf
        state["requires_confirmation"] = False
        print("[ConfirmRecovery] 用户取消")
    else:
        # 无法判断，清空 pending_confirmation 重新询问（路由到 confirm_node）
        state["pending_confirmation"] = {}
        state["requires_confirmation"] = True
        state["final_response"] = "请回复 是（确认）或 否（取消）。"
        state["response"] = state["final_response"]

    return state


# ============ 构建工作流 ============

def create_workflow(checkpointer=None):
    """创建 LangGraph 工作流

    架构说明：
    - check_profile → 档案检查
      - 档案不完整 → general_node（询问档案）
      - 档案完整 → init_daily_stats
    - init_daily_stats → classify_intent
    - classify_intent → routing_func（条件路由）
      - 单意图 → 对应节点 → END
      - 多意图 → fan-out 节点 → multi_join_node → END
      - confirm 意图 → confirm_node → confirm_recovery_node → commit_node → END
    """
    builder = StateGraph(AgentState)

    # === 节点 ===
    builder.add_node("check_profile", RouterAgent().check_profile)
    builder.add_node("init_daily_stats", init_daily_stats_node)
    builder.add_node("classify_intent", RouterAgent().classify_intent)
    builder.add_node("food_generate", food_generate_node)
    builder.add_node("workout_generate", workout_generate_node)
    builder.add_node("stats_node", stats_node)
    builder.add_node("recipe_node", recipe_node)
    builder.add_node("profile_node", profile_node)
    builder.add_node("general_node", general_node)
    builder.add_node("confirm_node", confirm_node)
    builder.add_node("confirm_recovery", confirm_recovery_node)
    builder.add_node("commit_node", commit_node)

    # === fan-out 多意图节点 ===
    builder.add_node("food_workout_fanout", food_workout_fanout)
    builder.add_node("food_stats_fanout", food_stats_fanout)
    builder.add_node("workout_stats_fanout", workout_stats_fanout)
    # 分支节点（由 Send 调度）
    builder.add_node("food_branch", food_branch)
    builder.add_node("workout_branch", workout_branch)
    builder.add_node("stats_branch", stats_branch)
    builder.add_node("multi_join_node", multi_join_node)

    # === 入口 ===
    builder.set_entry_point("check_profile")

    # 档案检查后路由
    builder.add_conditional_edges(
        "check_profile",
        profile_check_route,
        {
            "init_daily_stats": "init_daily_stats",
            "general_node": "general_node",
        }
    )

    # 初始化统计 → 意图分类
    builder.add_edge("init_daily_stats", "classify_intent")

    # 意图分类后 → 主路由
    builder.add_conditional_edges(
        "classify_intent",
        routing_func,
        {
            "food_generate": "food_generate",
            "workout_generate": "workout_generate",
            "stats_node": "stats_node",
            "recipe_node": "recipe_node",
            "profile_node": "profile_node",
            "general_node": "general_node",
            "confirm_node": "confirm_node",
            "food_workout_fanout": "food_workout_fanout",
            "food_stats_fanout": "food_stats_fanout",
            "workout_stats_fanout": "workout_stats_fanout",
        }
    )

    # === fan-out 节点的边 ===
    # fan-out 节点返回 [Send(...)] 调度分支，分支执行完后流向 join_node
    builder.add_edge("food_workout_fanout", "multi_join_node")
    builder.add_edge("food_stats_fanout", "multi_join_node")
    builder.add_edge("workout_stats_fanout", "multi_join_node")
    builder.add_edge("multi_join_node", END)

    # === confirm_node 的后续路由 ===
    # confirm_node 在 routing_func 层面处理了 Turn 1/Turn 2 区分
    # 这里直接结束本轮，等待用户下一条消息
    builder.add_edge("confirm_node", END)

    # confirm_recovery 路由：
    # - 用户确认/取消（confirmed=True/False）→ commit_node
    # - 用户回复不明确 → confirm_node 重新显示提示
    builder.add_conditional_edges(
        "confirm_recovery",
        lambda state: "commit_node" if state.get("pending_confirmation", {}).get("confirmed") is not None else "confirm_node",
        {"commit_node": "commit_node", "confirm_node": "confirm_node"}
    )
    builder.add_edge("commit_node", END)

    # === 单意图节点的确认路由 ===
    # 优先级：
    # 1. requires_confirmation=True + confirmed=True → 直接 commit_node（food_report / workout_report）
    # 2. requires_confirmation=True + confirmed=None → confirm_node（询问用户）
    # 3. requires_confirmation=False → END（无副作用的节点，如 general/stats/recipe）
    builder.add_conditional_edges(
        "food_generate",
        lambda state: (
            "commit_node"
            if state.get("requires_confirmation")
            and state.get("pending_confirmation", {}).get("confirmed") is True
            else ("confirm_node" if state.get("requires_confirmation") else END)
        ),
        {"confirm_node": "confirm_node", "commit_node": "commit_node", END: END}
    )

    builder.add_conditional_edges(
        "workout_generate",
        lambda state: (
            "commit_node"
            if state.get("requires_confirmation")
            and state.get("pending_confirmation", {}).get("confirmed") is True
            else ("confirm_node" if state.get("requires_confirmation") else END)
        ),
        {"confirm_node": "confirm_node", "commit_node": "commit_node", END: END}
    )

    # === 单意图直接结束 ===
    builder.add_edge("stats_node", END)
    builder.add_edge("recipe_node", END)
    builder.add_edge("profile_node", END)
    builder.add_edge("general_node", END)

    if checkpointer:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()
