"""
多意图并发处理节点 - 三层解耦架构

使用 LangGraph Send API 实现通用 fan-out / fan-in：
- generic_fanout: 根据 intent_plan.planned_branches 动态生成 Send
- 各分支写自己专属的字段（food_branch_result / workout_branch_result / stats_branch_result / recipe_branch_result）
- 各分支的 pending 上下文也写自己专属字段（food_pending_conf / workout_pending_conf）
- multi_join_node 读取分支专属字段，合成统一确认上下文

设计原则：
- 不再使用固定的 fan-out 组合（food_workout_fanout, food_stats_fanout 等）
- 通用 fan-out 根据 intent_plan 动态选择分支
- 支持任意组合的业务意图并行执行
"""

from langgraph.types import Send, Command
from agent.state import AgentState
from agent.food_agent import FoodAgent
from agent.workout_agent import WorkoutAgent
from agent.recipe_agent import RecipeAgent
from agent.memory import get_memory_agent

# ============ 常量定义 ============

# 分支节点名 → 意图过滤器映射（用于动态 fan-out）
BRANCH_INTENT_FILTERS = {
    "food_branch": {"food", "food_report"},
    "workout_branch": {"workout", "workout_report"},
    "stats_branch": {"stats_query"},
    "recipe_branch": {"recipe"},
}

# 分支结果字段映射
BRANCH_RESULT_FIELDS = {
    "food_branch": "food_branch_result",
    "workout_branch": "workout_branch_result",
    "stats_branch": "stats_branch_result",
    "recipe_branch": "recipe_branch_result",
}

# 分支 pending_conf 字段映射
BRANCH_PENDING_FIELDS = {
    "food_branch": "food_pending_conf",
    "workout_branch": "workout_pending_conf",
}

# 需要清理的字段（避免并发写入冲突）
_CLEANED_FIELDS = (
    "food_branch_result", "workout_branch_result", "stats_branch_result",
    "recipe_branch_result", "profile_branch_result",
    "food_pending_conf", "workout_pending_conf",
    "branch_results", "food_result", "workout_result", "stats_result", "recipe_result",
    "pending_confirmation", "requires_confirmation", "pending_stats",
)


# ============ 共享工具 ============

def _clear_state(state: dict) -> dict:
    """清除分支残留字段，避免并发写入冲突"""
    for key in _CLEANED_FIELDS:
        state.pop(key, None)
    return state


# ============ 通用 Fan-out ============

def generic_fanout(state: AgentState) -> Command:
    """通用 fan-out 节点 - 根据 intent_plan 动态生成 Send
    
    三层解耦的执行层：
    1. 从 intent_plan 读取 planned_branches
    2. 根据 intents 过滤每个分支的意图
    3. 动态生成 Send 列表
    """
    print(f"[Fanout] generic_fanout - 开始")
    state["route_decision"] = "generic_fanout"
    
    intent_plan = state.get("intent_plan")
    intents = state.get("intents", [state.get("intent", "general")])
    
    # 如果有 intent_plan，使用 planned_branches
    if intent_plan:
        planned_branches = intent_plan.get("planned_branches", [])
        print(f"[Fanout] generic_fanout - 使用 intent_plan: branches={planned_branches}")
    else:
        # 降级：根据 intents 动态推断分支
        planned_branches = _infer_branches_from_intents(intents)
        print(f"[Fanout] generic_fanout - 降级推断: branches={planned_branches}")
    
    # 动态生成 Send 列表
    sends = []
    for branch_name in planned_branches:
        intent_filter = BRANCH_INTENT_FILTERS.get(branch_name, set())
        # 过滤出属于该分支的意图
        filtered_intents = [i for i in intents if i in intent_filter]
        
        if filtered_intents:
            # 创建分支 state
            branch_state = _clear_state(dict(state))
            # 设置分支的 intent（归一化后）
            branch_state["intent"] = _normalize_intent(filtered_intents[0])
            sends.append(Send(branch_name, branch_state))
            print(f"[Fanout] generic_fanout - 添加分支: {branch_name} (intents={filtered_intents})")
    
    if not sends:
        # 无有效分支，返回 general_node
        print(f"[Fanout] generic_fanout - 无有效分支，降级到 general_node")
        return Command(goto="general_node")
    
    return Command(goto=sends)


def _infer_branches_from_intents(intents: list) -> list:
    """从 intents 推断分支列表（降级方案）"""
    branches = []
    for intent in intents:
        if intent in {"food", "food_report"}:
            if "food_branch" not in branches:
                branches.append("food_branch")
        elif intent in {"workout", "workout_report"}:
            if "workout_branch" not in branches:
                branches.append("workout_branch")
        elif intent == "stats_query":
            if "stats_branch" not in branches:
                branches.append("stats_branch")
        elif intent == "recipe":
            if "recipe_branch" not in branches:
                branches.append("recipe_branch")
    return branches


def _normalize_intent(intent: str) -> str:
    """归一化意图"""
    aliases = {"food_report": "food", "workout_report": "workout"}
    return aliases.get(intent, intent)


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


def recipe_branch(state: AgentState) -> AgentState:
    """Recipe 分支节点"""
    print(f"[Branch] recipe_branch 开始")
    recipe_agent = RecipeAgent()
    state = recipe_agent.run(state)
    return {
        "recipe_branch_result": state.get("response", ""),
        "recipe_result": state.get("response", ""),
    }


# ============ 汇合节点 ============

def multi_join_node(state: AgentState) -> dict:
    """
    多意图 fan-out 后的汇合节点。

    从分支专属字段读取各分支结果：
    - food_branch_result / workout_branch_result / stats_branch_result / recipe_branch_result
    - profile_branch_result（混合意图中的 profile_update 结果）
    - food_pending_conf / workout_pending_conf

    生成最终响应和统一的 pending_confirmation（用于 confirm 流程）。

    返回增量字段而非完整 state，防止 checkpoint 过大。
    """
    food_result = state.get("food_branch_result", "")
    workout_result = state.get("workout_branch_result", "")
    stats_result = state.get("stats_branch_result", "")
    recipe_result = state.get("recipe_branch_result", "")
    profile_result = state.get("profile_branch_result", "")
    food_conf = state.get("food_pending_conf") or {}
    workout_conf = state.get("workout_pending_conf") or {}

    print(f"[Join] multi_join_node - food_result={bool(food_result)}, "
          f"workout_result={bool(workout_result)}, stats_result={bool(stats_result)}, "
          f"recipe_result={bool(recipe_result)}, profile_result={bool(profile_result)}")

    # 合并各分支响应
    sections = []
    if food_result:
        sections.append(f"**食物记录**\n{food_result}")
    if workout_result:
        sections.append(f"**运动记录**\n{workout_result}")
    if stats_result:
        sections.append(f"**今日统计**\n{stats_result}")
    if recipe_result:
        sections.append(f"**推荐食谱**\n{recipe_result}")
    if profile_result:
        sections.append(f"**档案更新**\n{profile_result}")

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
        "recipe_branch_result": None,
        "profile_branch_result": None,
        "food_pending_conf": None,
        "workout_pending_conf": None,
        "food_result": None,
        "workout_result": None,
        "stats_result": None,
        "recipe_result": None,
    }



