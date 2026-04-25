from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Send, Command
from agent.state import AgentState
from agent.router_agent import RouterAgent, CONFIRM_WORDS, DENY_WORDS
from agent.food_agent import FoodAgent
from agent.workout_report_agent import WorkoutReportAgent
from agent.workout_advice_agent import WorkoutAdviceAgent
from agent.recipe_agent import RecipeAgent
from agent.planner import get_planner, decompose_tasks_node
from agent.multi_agent import (
    generic_fanout,
    food_branch, stats_branch, recipe_branch,
    workout_report_branch, workout_advice_branch,
    multi_join_node
)
from agent.memory import get_memory_agent
from agent.stream_utils import emit_trace
from config import AgentConfig
from datetime import datetime
import os


def log_node(node_name: str):
    """打印节点执行日志"""
    print(f"\n{'='*50}")
    print(f"[Graph] >>> 节点: {node_name}")
    print(f"{'='*50}\n")


# ============ Checkpointer ============

_postgres_cm = None
_postgres_async_cm = None


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


async def get_async_postgres_checkpointer():
    """创建异步 PostgreSQL checkpointer（用于 astream）

    使用 langgraph_checkpoint_postgres 的 AsyncPostgresSaver。
    必须异步初始化（__aenter__ 是异步的）。
    """
    global _postgres_async_cm
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    _postgres_async_cm = AsyncPostgresSaver.from_conn_string(db_url)
    saver = await _postgres_async_cm.__aenter__()
    await saver.setup()
    return saver


def get_memory_checkpointer():
    """创建内存 checkpointer（仅用于测试或临时使用）"""
    from langgraph.checkpoint.memory import InMemorySaver
    return InMemorySaver()


# 同步 checkpointer（用于 stream）
try:
    default_checkpointer = get_postgres_checkpointer()
except Exception as e:
    print(f"[Warning] PostgreSQL checkpointer init failed: {e}")
    print("[Warning] Falling back to InMemorySaver (state will be lost on restart)")
    default_checkpointer = get_memory_checkpointer()

# 异步 checkpointer（用于 astream）- 延迟初始化，需先 await 初始化
_async_checkpointer = None


async def get_async_default_checkpointer():
    """获取异步 checkpointer（延迟初始化，只初始化一次）"""
    global _async_checkpointer
    if _async_checkpointer is not None:
        return _async_checkpointer
    try:
        _async_checkpointer = await get_async_postgres_checkpointer()
        print("[Async Checkpointer] AsyncPostgresSaver initialized")
        return _async_checkpointer
    except Exception as e:
        print(f"[Warning] Async PostgreSQL checkpointer init failed: {e}")
        print("[Warning] Using InMemorySaver for astream (state will be lost on restart)")
        _async_checkpointer = get_memory_checkpointer()
        return _async_checkpointer


def trim_state(state: AgentState) -> None:
    """
    裁剪 state 中可能导致 checkpoint 过大的字段。
    在节点入口和 checkpoint 写入前调用。
    """
    # 1. messages 滑动窗口裁剪
    messages = state.get("messages", [])
    if len(messages) > AgentConfig.MAX_RECENT_MESSAGES:
        state["messages"] = messages[-AgentConfig.MAX_RECENT_MESSAGES:]

    # 2. summary_buffer 裁剪
    summary_buffer = state.get("summary_buffer", [])
    if len(summary_buffer) > AgentConfig.MAX_RECENT_MESSAGES:
        state["summary_buffer"] = summary_buffer[-AgentConfig.MAX_RECENT_MESSAGES:]

    # 3. 清空 image_info（base64 data URL 已在节点内使用完毕，无需持久化）
    image_info = state.get("image_info", {})
    if image_info.get("image_url"):
        state["image_info"] = {"has_image": bool(image_info.get("has_image")), "image_url": ""}

    # 4. 清空过大的 response/final_response（流式输出后不再需要完整副本）
    for key in ("response", "final_response"):
        val = state.get(key, "")
        if isinstance(val, str) and len(val) > 5000:
            state[key] = val[:5000] + "...(trimmed)"

    # 5. 清空 agent 原始结果（join 后不再需要）
    for key in ("food_result", "workout_result", "stats_result", "recipe_result"):
        state.pop(key, None)
    
    # 6. 清空分支结果字段（fan-out join 后不再需要）
    for key in ("food_branch_result", "stats_branch_result",
                "recipe_branch_result", "profile_branch_result",
                "workout_report_branch_result", "workout_advice_branch_result",
                "food_pending_conf",
                "workout_report_pending_conf", "workout_advice_pending_conf"):
        state.pop(key, None)


# ============ 路由函数 ============

# 意图 → 节点名 映射（单意图，用于直接路由）
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
    """主路由函数 - 基于 intent_plan 动态路由
    
    三层解耦架构：
    1. classify_intent 输出 intents（只负责"看懂"）
    2. intent_planner 生成 intent_plan（负责规划）
    3. routing_func 根据 intent_plan 路由（负责执行调度）
    
    返回值：
    - 单意图：返回节点名字符串
    - 多意图：返回 "generic_fanout" 节点（由该节点动态生成 Send）
    - confirm 特殊处理：根据 pending_confirmation 状态路由
    """
    intent_plan = state.get("intent_plan")
    
    # 如果没有 intent_plan（兼容旧逻辑），使用 intents 直接路由
    if not intent_plan:
        print(f"[Router] routing_func - 无 intent_plan，降级到旧逻辑")
        return _legacy_routing_func(state)
    
    print(f"[Router] routing_func - intent_plan: mode={intent_plan['execution_mode']}, "
          f"branches={intent_plan['planned_branches']}")
    
    # ---- 特殊意图优先处理 ----
    special_intents = intent_plan.get("special_intents", [])
    regular_intents = intent_plan.get("regular_intents", [])
    
    # 纯特殊意图：单独处理
    if not regular_intents and special_intents:
        intent = special_intents[0]
        if intent == "confirm":
            pending_conf = state.get("pending_confirmation") or {}
            if pending_conf and pending_conf.get("confirmed") is None:
                return "confirm_recovery"
            return "confirm_node"
        elif intent == "profile_update":
            return "profile_node"
    
    # ---- 混合意图：特殊意图 + 业务意图 ----
    # profile_update + regular intents：先执行 profile_update，再执行业务意图
    if special_intents and regular_intents:
        print(f"[Router] routing_func - 混合意图: special={special_intents}, regular={regular_intents}")
        if "profile_update" in special_intents:
            # 先执行 profile_update，然后通过条件边路由到 generic_fanout
            print(f"[Router] routing_func - 混合意图含 profile_update → profile_update_pre")
            return "profile_update_pre"
        # 其他特殊意图（如 confirm）需要特殊处理
        # confirm 混合业务意图的场景暂不支持，降级到 general_node
        if "confirm" in special_intents:
            print(f"[Router] routing_func - confirm 混合业务意图暂不支持 → general_node")
            return "general_node"
    
    # ---- 单意图路由 ----
    if intent_plan["execution_mode"] == "single":
        primary = intent_plan["primary_intent"]
        node = INTENT_TO_NODE.get(primary)
        if node:
            print(f"[Router] routing_func - 单意图 {primary} → {node}")
            return node
        print(f"[Router] routing_func - 单意图 {primary} 无对应节点 → general_node")
        return "general_node"
    
    # ---- 多意图 fan-out ----
    # 使用通用 fan-out 节点，动态根据 planned_branches 生成 Send
    if intent_plan["execution_mode"] == "parallel":
        planned_branches = intent_plan.get("planned_branches", [])
        if planned_branches:
            print(f"[Router] routing_func - 多意图 parallel → generic_fanout")
            return "generic_fanout"
        # 无有效分支，降级
        print(f"[Router] routing_func - 多意图无有效分支 → general_node")
        return "general_node"
    
    # 兜底
    print(f"[Router] routing_func - 未知 execution_mode → general_node")
    return "general_node"


def _legacy_routing_func(state: AgentState):
    """旧版路由函数 - 兼容没有 intent_plan 的情况"""
    intents = state.get("intents", [state.get("intent", "general")])
    print(f"[Router] _legacy_routing_func - intents: {intents}")
    
    intents = [i for i in intents if i]
    if not intents:
        return "general_node"
    
    # 归一化
    _intent_aliases = {"food_report": "food", "workout_report": "workout"}
    normalized = [_intent_aliases.get(i, i) for i in intents]
    intents = normalized
    
    # 单意图
    if len(intents) == 1:
        node = INTENT_TO_NODE.get(intents[0])
        return node if node else "general_node"
    
    # 多意图 - 降级到通用 fan-out
    return "generic_fanout"


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
    emit_trace("node_start", "init_daily_stats", "正在加载今日统计...")
    # 入图时压缩 checkpoint 体积，防止 PostgreSQL 单条记录超限
    _cap_checkpoint_size(state)
    state["route_decision"] = "init_daily_stats"
    memory_agent = get_memory_agent()
    today = datetime.now().strftime("%Y-%m-%d")
    stats = memory_agent.load_daily_stats(today)
    state["daily_stats"] = stats
    emit_trace("node_end", "init_daily_stats", "执行完成")
    return state


def intent_planner_node(state: AgentState) -> AgentState:
    """意图规划节点 - 三层解耦的规划层
    
    将 classify_intent 输出的 intents 转换为执行计划。
    执行计划包含：执行模式、分支列表、特殊意图处理方式等。
    """
    log_node("intent_planner")
    planner = get_planner()
    state = planner.plan(state)
    return state


def food_generate_node(state: AgentState) -> AgentState:
    """食物分析 + 生成候选结果节点"""
    log_node("food_generate")
    emit_trace("node_start", "food_generate", "正在分析食物营养成分...")
    state["route_decision"] = "food_generate"
    food_agent = FoodAgent()
    state = food_agent.run(state)
    emit_trace("node_end", "food_generate", "执行完成")
    trim_state(state)
    return state


def workout_generate_node(state: AgentState) -> AgentState:
    """运动节点（单意图路径）- 根据 intent 选择记录或咨询 agent"""
    log_node("workout_generate")
    emit_trace("node_start", "workout_generate", "正在生成运动方案...")
    state["route_decision"] = "workout_generate"

    intent = state.get("intent", "workout")
    if intent == "workout_report":
        agent = WorkoutReportAgent()
    else:
        agent = WorkoutAdviceAgent()
    state = agent.run(state)

    emit_trace("node_end", "workout_generate", "执行完成")
    trim_state(state)
    return state


def stats_node(state: AgentState) -> AgentState:
    """统计查询节点"""
    log_node("stats_node")
    emit_trace("node_start", "stats_node", "正在查询今日统计...")
    state["route_decision"] = "stats_node"
    memory_agent = get_memory_agent()
    summary = memory_agent.get_daily_summary()
    state["final_response"] = summary
    state["response"] = summary
    emit_trace("node_end", "stats_node", "执行完成")
    trim_state(state)
    return state


def recipe_node(state: AgentState) -> AgentState:
    """食谱推荐节点"""
    log_node("recipe_node")
    emit_trace("node_start", "recipe_node", "正在生成推荐食谱...")
    state["route_decision"] = "recipe_node"
    recipe_agent = RecipeAgent()
    state = recipe_agent.run(state)
    state["final_response"] = state.get("response")
    emit_trace("node_end", "recipe_node", "执行完成")
    trim_state(state)
    return state


def profile_node(state: AgentState) -> AgentState:
    """档案更新节点"""
    log_node("profile_node")
    emit_trace("node_start", "profile_node", "正在更新用户档案...")
    state["route_decision"] = "profile_node"
    router = RouterAgent()
    state = router.handle_profile_update(state)
    state["final_response"] = state.get("response")
    emit_trace("node_end", "profile_node", "执行完成")
    trim_state(state)
    return state


def profile_update_pre_node(state: AgentState) -> AgentState:
    """档案更新预处理节点 - 用于混合意图场景
    
    当 profile_update 与 business intents 混合时：
    1. 先执行 profile_update 逻辑
    2. 保存结果到 profile_branch_result
    3. 然后通过条件边路由到 generic_fanout 执行其他业务意图
    
    这样 profile_update 不会被静默丢弃。
    """
    log_node("profile_update_pre")
    emit_trace("node_start", "profile_update_pre", "正在更新用户档案...")
    state["route_decision"] = "profile_update_pre"
    router = RouterAgent()
    state = router.handle_profile_update(state)
    # 保存 profile_update 结果，供 multi_join_node 合并
    state["profile_branch_result"] = state.get("response", "")
    emit_trace("node_end", "profile_update_pre", "档案更新完成")
    trim_state(state)
    return state


def general_node(state: AgentState) -> AgentState:
    """通用对话节点"""
    log_node("general_node")
    emit_trace("node_start", "general_node", "正在思考回复...")
    state["route_decision"] = "general_node"
    router = RouterAgent()
    state = router.handle_general(state)
    state["final_response"] = state.get("response")
    emit_trace("node_end", "general_node", "执行完成")
    trim_state(state)
    return state


def confirm_node(state: AgentState) -> AgentState:
    """确认节点 - 展示待确认内容并等待用户回复

    首次进入：设置 pending_confirmation，显示确认提示
    routing_func 根据 pending_confirmation.confirmed 是否为 None 判断是否在等待回复

    标题渲染规则（方案A：后端统一负责标题）：
    - 单意图确认：本节点添加标题
    - 多意图确认（log_both）：不在此添加标题，因为 multi_join_node 已经添加过了
    - 多意图场景下，优先从 food_branch_result / workout_branch_result 读取内容
    """
    log_node("confirm_node")
    emit_trace("node_start", "confirm_node", "正在处理确认请求...")
    state["route_decision"] = "confirm_node"

    pending_conf = state.get("pending_confirmation") or {}
    action = pending_conf.get("action")
    candidate_meal = pending_conf.get("candidate_meal")
    candidate_workout = pending_conf.get("candidate_workout")

    # 辅助函数：按优先级读取分支结果
    # 多意图场景下 branch_result 字段有值，单意图场景下 food_result / workout_result 有值
    def _read_food_analysis(state_dict):
        """优先读取 food_branch_result（多意图），回退到 food_result（单意图）"""
        return (
            state_dict.get("food_branch_result")
            or state_dict.get("food_result")
            or state_dict.get("response")
            or ""
        )

    def _read_workout_analysis(state_dict):
        """优先读取 workout 分支结果（多意图），回退到 workout_result（单意图）"""
        return (
            state_dict.get("workout_report_branch_result")
            or state_dict.get("workout_advice_branch_result")
            or state_dict.get("workout_result")
            or state_dict.get("response")
            or ""
        )

    # 设置 pending_confirmation（confirmed=None 表示"等待回复"）
    # routing_func 根据 pending_confirmation 是否存在来判断：
    # - 有数据且 confirmed=None → 用户在回复确认提示 → routing_func 会路由到 confirm_recovery
    # - 无数据 → 新确认周期开始 → 显示确认提示
    if action == "log_meal" and candidate_meal:
        analysis = _read_food_analysis(state)
        state["pending_confirmation"] = {
            "action": "log_meal",
            "candidate_meal": candidate_meal,
            "candidate_workout": None,
            "analysis_text": "（待确认食物记录）",
            "confirmed": None,
        }
        state["final_response"] = (
            f"{analysis}\n\n---\n"
            f"是否将上述食物计入今日热量统计？（是/否）"
        )
    elif action == "log_workout" and candidate_workout:
        analysis = _read_workout_analysis(state)
        state["pending_confirmation"] = {
            "action": "log_workout",
            "candidate_meal": None,
            "candidate_workout": candidate_workout,
            "analysis_text": "（待确认运动记录）",
            "confirmed": None,
        }
        state["final_response"] = (
            f"{analysis}\n\n---\n"
            f"是否将上述运动计入今日消耗统计？（是/否）"
        )
    elif action == "log_both":
        # 多意图确认：读取分支结果字段，不重复添加标题
        # （标题已在 multi_join_node 中统一添加）
        food_analysis = _read_food_analysis(state)
        workout_analysis = _read_workout_analysis(state)
        state["pending_confirmation"] = {
            "action": "log_both",
            "candidate_meal": candidate_meal,
            "candidate_workout": candidate_workout,
            "analysis_text": "（含食物和运动记录，请确认）",
            "confirmed": None,
        }
        # 只有当分支结果为空时（单意图 fallback），才添加标题
        # 多意图场景下 analysis 内容已包含标题，直接拼接确认提示
        if food_analysis and workout_analysis:
            # 多意图：标题已在 multi_join_node 中添加，不重复
            state["final_response"] = (
                f"{food_analysis}\n\n{workout_analysis}\n\n---\n"
                f"是否将上述记录计入今日统计？（是/否）"
            )
        elif food_analysis:
            state["final_response"] = (
                f"**食物记录**\n{food_analysis}\n\n---\n"
                f"是否将上述食物计入今日热量统计？（是/否）"
            )
        elif workout_analysis:
            state["final_response"] = (
                f"**运动记录**\n{workout_analysis}\n\n---\n"
                f"是否将上述运动计入今日消耗统计？（是/否）"
            )
        else:
            state["final_response"] = (
                f"是否将上述记录计入今日统计？（是/否）"
            )
    else:
        # 没有待确认内容
        state["final_response"] = "没有待确认的记录。"
        state["requires_confirmation"] = False

    state["response"] = state["final_response"]
    emit_trace("node_end", "confirm_node", "执行完成")
    trim_state(state)
    return state


def commit_node(state: AgentState) -> AgentState:
    """提交节点 - 将确认后的记录写入 daily_stats"""
    log_node("commit_node")
    emit_trace("node_start", "commit_node", "正在提交记录...")
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
        # 保留 LLM 生成的分析/建议内容（如有），避免 commit 覆盖掉回答
        analysis_text = pending_conf.get("analysis_text", "")
        if analysis_text and analysis_text not in ("（待确认食物记录）", "（待确认运动记录）", "（含食物和运动记录，请确认）", "（请确认是否计入）"):
            state["final_response"] = f"{analysis_text}\n\n---\n已计入统计。\n\n{summary}"
        else:
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
    state["pending_confirmation"] = {}

    # --- legacy 兼容层清理（pending_stats.json fallback）---
    state["pending_stats"] = None
    try:
        memory_agent.clear_pending_stats()
    except Exception:
        pass

    emit_trace("node_end", "commit_node", "执行完成")
    trim_state(state)
    return state


def confirm_recovery_node(state: AgentState) -> AgentState:
    """确认恢复节点 - 处理用户对 confirm_node 问题的回复

    当用户回复"是/好/确认"或"取消/不算"时，这个节点读取回复内容，
    设置 pending_confirmation.confirmed 标志，然后流向 commit_node。
    """
    log_node("confirm_recovery")
    emit_trace("node_start", "confirm_recovery", "正在处理确认回复...")
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

    emit_trace("node_end", "confirm_recovery", "执行完成")
    trim_state(state)
    return state


# ============ 构建工作流 ============

def create_workflow(checkpointer=None):
    """创建 LangGraph 工作流
    
    三层解耦架构：
    1. classify_intent: 意图识别层（只负责"看懂"）
    2. intent_planner: 意图规划层（生成执行计划）
    3. routing_func: 执行调度层（根据计划路由）
    
    工作流程：
    check_profile → 档案检查
      - 档案不完整 → general_node（询问档案）
      - 档案完整 → init_daily_stats
    init_daily_stats → classify_intent → intent_planner → routing_func
      - 单意图 → 对应节点 → END
      - 多意图 → generic_fanout → multi_join_node → END
      - confirm 意图 → confirm_node → confirm_recovery_node → commit_node → END
    """
    builder = StateGraph(AgentState)

    # === 节点 ===
    builder.add_node("check_profile", RouterAgent().check_profile)
    builder.add_node("init_daily_stats", init_daily_stats_node)
    builder.add_node("classify_intent", RouterAgent().classify_intent)
    builder.add_node("decompose_tasks", decompose_tasks_node)
    builder.add_node("intent_planner", intent_planner_node)
    builder.add_node("food_generate", food_generate_node)
    builder.add_node("workout_generate", workout_generate_node)
    builder.add_node("stats_node", stats_node)
    builder.add_node("recipe_node", recipe_node)
    builder.add_node("profile_node", profile_node)
    builder.add_node("profile_update_pre", profile_update_pre_node)  # 混合意图中的 profile_update 预处理
    builder.add_node("general_node", general_node)
    builder.add_node("confirm_node", confirm_node)
    builder.add_node("confirm_recovery", confirm_recovery_node)
    builder.add_node("commit_node", commit_node)

    # === 通用 fan-out 节点 ===
    builder.add_node("generic_fanout", generic_fanout)
    # 分支节点（由 Send 调度）
    builder.add_node("food_branch", food_branch)
    builder.add_node("workout_report_branch", workout_report_branch)
    builder.add_node("workout_advice_branch", workout_advice_branch)
    builder.add_node("stats_branch", stats_branch)
    builder.add_node("recipe_branch", recipe_branch)
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

    # 初始化统计 → 意图分类 → 子任务分解 → 意图规划
    builder.add_edge("init_daily_stats", "classify_intent")
    builder.add_edge("classify_intent", "decompose_tasks")
    builder.add_edge("decompose_tasks", "intent_planner")

    # 意图规划后 → 主路由
    builder.add_conditional_edges(
        "intent_planner",
        routing_func,
        {
            "food_generate": "food_generate",
            "workout_generate": "workout_generate",
            "stats_node": "stats_node",
            "recipe_node": "recipe_node",
            "profile_node": "profile_node",
            "profile_update_pre": "profile_update_pre",  # 混合意图中的 profile_update 预处理
            "general_node": "general_node",
            "confirm_node": "confirm_node",
            "confirm_recovery": "confirm_recovery",
            "generic_fanout": "generic_fanout",
        }
    )

    # === fan-out 节点的边 ===
    # generic_fanout 通过 Send 动态调度分支；
    # join 节点只在各分支完成后由分支边触发，避免抢跑
    builder.add_edge("food_branch", "multi_join_node")
    builder.add_edge("workout_report_branch", "multi_join_node")
    builder.add_edge("workout_advice_branch", "multi_join_node")
    builder.add_edge("stats_branch", "multi_join_node")
    builder.add_edge("recipe_branch", "multi_join_node")
    builder.add_edge("multi_join_node", END)

    # profile_update_pre 执行完后路由到 generic_fanout 执行业务意图
    builder.add_edge("profile_update_pre", "generic_fanout")

    # === confirm_node 的后续路由 ===
    # confirm_node 在 routing_func 层面处理了 Turn 1/Turn 2 区分
    # 这里直接结束本轮，等待用户下一条消息
    builder.add_edge("confirm_node", END)

    # confirm_recovery 路由：
    # - 用户确认/取消（confirmed=True/False）→ commit_node
    # - 用户回复不明确 → confirm_node 重新显示提示
    def _confirm_recovery_route(state):
        pending_conf = state.get("pending_confirmation") or {}
        return "commit_node" if pending_conf.get("confirmed") is not None else "confirm_node"
    
    builder.add_conditional_edges(
        "confirm_recovery",
        _confirm_recovery_route,
        {"commit_node": "commit_node", "confirm_node": "confirm_node"}
    )
    builder.add_edge("commit_node", END)

    # === 单意图节点的确认路由 ===
    # 优先级：
    # 1. requires_confirmation=True + confirmed=True → 直接 commit_node（food_report / workout_report）
    # 2. requires_confirmation=True + confirmed=None → confirm_node（询问用户）
    # 3. requires_confirmation=False → END（无副作用的节点，如 general/stats/recipe）
    def _food_generate_route(state):
        if not state.get("requires_confirmation"):
            return END
        pending_conf = state.get("pending_confirmation") or {}
        return "commit_node" if pending_conf.get("confirmed") is True else "confirm_node"
    
    builder.add_conditional_edges(
        "food_generate",
        _food_generate_route,
        {"confirm_node": "confirm_node", "commit_node": "commit_node", END: END}
    )

    def _workout_generate_route(state):
        if not state.get("requires_confirmation"):
            return END
        pending_conf = state.get("pending_confirmation") or {}
        return "commit_node" if pending_conf.get("confirmed") is True else "confirm_node"
    
    builder.add_conditional_edges(
        "workout_generate",
        _workout_generate_route,
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
