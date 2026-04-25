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
from agent.workout_report_agent import WorkoutReportAgent
from agent.workout_advice_agent import WorkoutAdviceAgent
from agent.recipe_agent import RecipeAgent
from agent.memory import get_memory_agent

# ============ 常量定义 ============

# 分支节点名 → 意图过滤器映射（用于动态 fan-out）
BRANCH_INTENT_FILTERS = {
    "food_branch": {"food", "food_report"},
    "workout_report_branch": {"workout_report"},
    "workout_advice_branch": {"workout", "recovery"},
    "stats_branch": {"stats_query"},
    "recipe_branch": {"recipe"},
}

# 需要清理的字段（避免并发写入冲突）
_CLEANED_FIELDS = (
    "food_branch_result", "stats_branch_result",
    "recipe_branch_result", "profile_branch_result",
    "workout_report_branch_result", "workout_advice_branch_result",
    "food_pending_conf",
    "workout_report_pending_conf", "workout_advice_pending_conf",
    "branch_results", "food_result", "workout_result", "stats_result", "recipe_result",
    "pending_confirmation", "requires_confirmation", "pending_stats",
    "source_intents",
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
    3. 动态生成 Send 列表，同时传递分支专属的 prompt bundle
    """
    print(f"[Fanout] generic_fanout - 开始")
    state["route_decision"] = "generic_fanout"
    
    intent_plan = state.get("intent_plan")
    intents = state.get("intents", [state.get("intent", "general")])
    branch_bundles = state.get("branch_prompt_bundles") or {}
    
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
        # 过滤出属于该分支的意图（保留原始形式，如 food_report / workout_report）
        filtered_intents = [i for i in intents if i in intent_filter]
        
        if filtered_intents:
            # 创建分支 state
            branch_state = _clear_state(dict(state))
            # 设置分支的 intent（归一化后，用于路由）
            branch_state["intent"] = _normalize_intent(filtered_intents[0])
            # 保留原始意图列表（用于 agent 内判断 reporting 语义）
            branch_state["source_intents"] = list(filtered_intents)
            
            # 传递该分支的 prompt bundle（如果存在）
            if branch_name in branch_bundles:
                branch_state["branch_prompt_bundle"] = branch_bundles[branch_name]
                print(f"[Fanout] generic_fanout - 传递 {branch_name} prompt bundle")
            
            sends.append(Send(branch_name, branch_state))
            print(f"[Fanout] generic_fanout - 添加分支: {branch_name} (intents={filtered_intents}, source_intents={filtered_intents})")
    
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
        elif intent == "workout":
            if "workout_advice_branch" not in branches:
                branches.append("workout_advice_branch")
        elif intent == "workout_report":
            if "workout_report_branch" not in branches:
                branches.append("workout_report_branch")
        elif intent == "recovery":
            if "workout_advice_branch" not in branches:
                branches.append("workout_advice_branch")
        elif intent == "stats_query":
            if "stats_branch" not in branches:
                branches.append("stats_branch")
        elif intent == "recipe":
            if "recipe_branch" not in branches:
                branches.append("recipe_branch")
    return branches


def _normalize_intent(intent: str) -> str:
    """归一化意图（保留 report 语义，不合并）"""
    return intent


# ============ 分支节点 ============

def food_branch(state: AgentState) -> AgentState:
    """Food 分支节点（由 Send 并发调度）
    
    消费 planner 生成的 branch_prompt_bundle，避免自己重新编排全局上下文。
    """
    print(f"[Branch] food_branch 开始")
    
    # 检查是否有预生成的 prompt bundle
    bundle = state.get("branch_prompt_bundle")
    if bundle:
        print(f"[Branch] food_branch - 使用预生成的 prompt bundle")
        extra_sections = bundle.get("extra_sections", {})
        branch_input = bundle.get("branch_input", "")
    else:
        # Fallback: 让 agent 自己构建
        extra_sections = None
        branch_input = None
    
    food_agent = FoodAgent()
    state = food_agent.run(
        state,
        extra_sections=extra_sections,
        branch_input=branch_input,
        append_history=False,
    )
    return {
        "food_branch_result": state.get("food_result") or "",
        "food_pending_conf": state.get("pending_confirmation"),
    }


def workout_report_branch(state: AgentState) -> AgentState:
    """运动记录分支节点（由 Send 并发调度）

    只处理 workout_report 意图：提取运动数据、设置确认。
    """
    print(f"[Branch] workout_report_branch 开始")

    bundle = state.get("branch_prompt_bundle")
    if bundle:
        print(f"[Branch] workout_report_branch - 使用预生成的 prompt bundle")
        extra_sections = bundle.get("extra_sections", {})
        branch_input = bundle.get("branch_input", "")
    else:
        extra_sections = None
        branch_input = None

    agent = WorkoutReportAgent()
    state = agent.run(
        state,
        extra_sections=extra_sections,
        branch_input=branch_input,
        append_history=False,
    )
    return {
        "workout_report_branch_result": state.get("workout_result") or "",
        "workout_report_pending_conf": state.get("pending_confirmation"),
    }


def workout_advice_branch(state: AgentState) -> AgentState:
    """运动咨询分支节点（由 Send 并发调度）

    只处理 workout 意图（咨询）：检索知识、生成建议。
    """
    print(f"[Branch] workout_advice_branch 开始")

    bundle = state.get("branch_prompt_bundle")
    if bundle:
        print(f"[Branch] workout_advice_branch - 使用预生成的 prompt bundle")
        extra_sections = bundle.get("extra_sections", {})
        branch_input = bundle.get("branch_input", "")
    else:
        extra_sections = None
        branch_input = None

    agent = WorkoutAdviceAgent()
    state = agent.run(
        state,
        extra_sections=extra_sections,
        branch_input=branch_input,
        append_history=False,
    )
    return {
        "workout_advice_branch_result": state.get("workout_result") or "",
        "workout_advice_pending_conf": state.get("pending_confirmation"),
    }


def stats_branch(state: AgentState) -> AgentState:
    """Stats 分支节点
    
    消费 planner 生成的 branch_prompt_bundle，使用预格式化的统计上下文。
    """
    print(f"[Branch] stats_branch 开始")
    
    # 检查是否有预生成的 prompt bundle
    bundle = state.get("branch_prompt_bundle")
    if bundle and bundle.get("branch_context"):
        print(f"[Branch] stats_branch - 使用预生成的 branch_context")
        summary = bundle.get("branch_context", "")
    else:
        memory_agent = get_memory_agent()
        summary = memory_agent.get_daily_summary()

    return {
        "stats_branch_result": summary or "",
        "stats_result": summary or "",
    }


def recipe_branch(state: AgentState) -> AgentState:
    """Recipe 分支节点
    
    消费 planner 生成的 branch_prompt_bundle，直接使用预计算的营养约束。
    """
    print(f"[Branch] recipe_branch 开始")
    
    # 检查是否有预生成的 prompt bundle
    bundle = state.get("branch_prompt_bundle")
    if bundle:
        print(f"[Branch] recipe_branch - 使用预生成的 prompt bundle")
        extra_sections = bundle.get("extra_sections", {})
        branch_input = bundle.get("branch_input", "")
    else:
        # Fallback: 让 agent 自己构建
        extra_sections = None
        branch_input = None
    
    recipe_agent = RecipeAgent()
    state = recipe_agent.run(
        state,
        extra_sections=extra_sections,
        branch_input=branch_input,
        append_history=False,
    )
    return {
        "recipe_branch_result": state.get("response") or "",
        "recipe_result": state.get("response") or "",
    }


# ============ 汇合节点 ============

def multi_join_node(state: AgentState) -> dict:
    """
    多意图 fan-out 后的汇合节点。

    从分支专属字段读取各分支结果：
    - food_branch_result / stats_branch_result / recipe_branch_result
    - workout_report_branch_result / workout_advice_branch_result
    - profile_branch_result（混合意图中的 profile_update 结果）
    - food_pending_conf / workout_report_pending_conf

    生成最终响应和统一的 pending_confirmation（用于 confirm 流程）。

    返回增量字段而非完整 state，防止 checkpoint 过大。
    """
    food_result = state.get("food_branch_result", "")
    workout_report_result = state.get("workout_report_branch_result", "")
    workout_advice_result = state.get("workout_advice_branch_result", "")
    stats_result = state.get("stats_branch_result", "")
    recipe_result = state.get("recipe_branch_result", "")
    profile_result = state.get("profile_branch_result", "")
    food_conf = state.get("food_pending_conf") or {}
    workout_report_conf = state.get("workout_report_pending_conf") or {}
    intent_plan = state.get("intent_plan") or {}
    expected_branches = intent_plan.get("planned_branches", [])

    # 合并运动相关结果
    effective_workout_result = workout_report_result or workout_advice_result
    effective_workout_conf = workout_report_conf

    # 详细日志：记录 join 节点收到的所有分支结果状态
    print(f"[Join] multi_join_node - food_result={bool(food_result)}, "
          f"workout_report_result={bool(workout_report_result)}, "
          f"workout_advice_result={bool(workout_advice_result)}, "
          f"stats_result={bool(stats_result)}, "
          f"recipe_result={bool(recipe_result)}, profile_result={bool(profile_result)}")

    branch_value_map = {
        "food_branch": food_result,
        "workout_report_branch": workout_report_result,
        "workout_advice_branch": workout_advice_result,
        "stats_branch": stats_result,
        "recipe_branch": recipe_result,
    }
    expected_nonempty = [
        branch for branch in expected_branches
        if branch in branch_value_map
    ]
    ready_branches = [
        branch for branch in expected_nonempty
        if branch_value_map.get(branch) is not None
    ]

    if expected_nonempty and len(ready_branches) < len(expected_nonempty):
        pending = [branch for branch in expected_nonempty if branch not in ready_branches]
        print(
            f"[Join] multi_join_node - 分支未完成，等待: {pending}，"
            f"current keys: {list(state.keys())}"
        )
        return {
            "route_decision": "multi_join_wait",
        }

    # 检查是否有任何分支结果
    has_any_result = any([food_result, effective_workout_result, stats_result, recipe_result, profile_result])

    # 合并各分支响应
    # 标题由 multi_join_node 统一负责（方案A），其他节点不得重复添加
    sections = []
    if food_result:
        sections.append(f"**食物记录**\n{food_result}")
    if workout_report_result and workout_advice_result:
        # 两个分支都有结果：分别显示
        sections.append(f"**运动记录**\n{workout_report_result}")
        sections.append(f"**运动建议**\n{workout_advice_result}")
    elif workout_report_result:
        # 只有记录分支
        sections.append(f"**运动记录**\n{workout_report_result}")
    elif workout_advice_result:
        # 只有咨询分支
        sections.append(f"**运动建议**\n{workout_advice_result}")
    if stats_result:
        sections.append(f"**今日统计**\n{stats_result}")
    if recipe_result:
        sections.append(f"**推荐食谱**\n{recipe_result}")
    if profile_result:
        sections.append(f"**档案更新**\n{profile_result}")

    # 只有当有分支结果时才合并；空字符串不应被当作有效结果
    # 注意：fallback "已处理您的请求。" 绝不作为 food/workout 分支的正文输出
    if not sections:
        if not has_any_result:
            print(f"[Join] WARNING: 所有分支结果均为空！state keys: {list(state.keys())}")
        final_response = "抱歉，处理您的请求时出现问题，请重试。"
    else:
        final_response = "\n\n".join(sections)

    # 确定 pending_confirmation（analysis_text 只保留摘要，不保留全文）
    unified_conf = None
    has_food = bool(food_conf.get("action"))
    has_workout = bool(effective_workout_conf.get("action"))

    if has_food and has_workout:
        unified_conf = {
            "action": "log_both",
            "candidate_meal": food_conf.get("candidate_meal"),
            "candidate_workout": effective_workout_conf.get("candidate_workout"),
            "analysis_text": final_response,
            "confirmed": None,
        }
    elif has_food:
        unified_conf = {
            "action": food_conf.get("action"),
            "candidate_meal": food_conf.get("candidate_meal"),
            "candidate_workout": None,
            "analysis_text": final_response,
            "confirmed": None,
        }
    elif has_workout:
        unified_conf = {
            "action": effective_workout_conf.get("action"),
            "candidate_meal": None,
            "candidate_workout": effective_workout_conf.get("candidate_workout"),
            "analysis_text": final_response,
            "confirmed": None,
        }

    if unified_conf:
        final_response += "\n\n---\n是否将上述记录计入今日统计？（是/否）"

    # 只返回 join 节点需要写入的字段
    # 不要返回清理字段（如 food_branch_result=None），这会与 branch 节点在同一 superstep 冲突
    # 清理工作交给 graph 完成后的 trim_state 处理
    return {
        "route_decision": "multi_join_node",
        "final_response": final_response,
        "response": final_response,
        "pending_confirmation": unified_conf,
        "requires_confirmation": bool(unified_conf),
    }

