"""
意图规划层 - 三层解耦架构的核心

职责：
1. 意图归一化（food_report → food, workout_report → workout）
2. 特殊意图与业务意图分离
3. 生成执行计划（执行模式、分支列表）

设计原则：
- 识别层只负责"看懂"，不决定路由
- 规划层把意图集合转换为执行计划
- 执行层只负责跑各自分支，最后统一合并

特殊意图处理：
- confirm: 会话控制层，独立处理，但不吞掉业务意图
- profile_update: 作为 overlay，不替代 food/workout/recipe/stats_query
- general: 降级处理，当无有效业务意图时使用

业务意图（可并行）：
- food / food_report → food_branch
- workout / workout_report → workout_branch
- recipe → recipe_branch
- stats_query → stats_branch
"""

from agent.state import AgentState, IntentPlan
from agent.stream_utils import emit_trace

# ============ 常量定义 ============

# 特殊意图（会话控制层）
SPECIAL_INTENTS = {"confirm", "profile_update"}

# 业务意图（可并行执行）
BUSINESS_INTENTS = {"food", "workout", "recipe", "stats_query"}

# 意图别名映射
INTENT_ALIASES = {
    "food_report": "food",
    "workout_report": "workout",
}

# 意图 → 分支节点名映射
INTENT_TO_BRANCH = {
    "food": "food_branch",
    "workout": "workout_branch",
    "recipe": "recipe_branch",
    "stats_query": "stats_branch",
}


# ============ 意图规划器 ============

class IntentPlanner:
    """意图规划器 - 把意图集合转换为执行计划"""
    
    def __init__(self):
        pass
    
    def plan(self, state: AgentState) -> AgentState:
        """规划节点 - 在 classify_intent 之后执行，生成执行计划"""
        print(f"[Planner] plan - 开始意图规划")
        emit_trace("node_start", "intent_planner", "正在规划执行计划...")
        
        intents = state.get("intents", [state.get("intent", "general")])
        print(f"[Planner] plan - 输入意图: {intents}")
        
        # 1. 意图归一化
        normalized_intents = self._normalize_intents(intents)
        print(f"[Planner] plan - 归一化后: {normalized_intents}")
        
        # 2. 分离特殊意图和业务意图
        special_intents, regular_intents = self._separate_intents(normalized_intents)
        print(f"[Planner] plan - 特殊意图: {special_intents}, 业务意图: {regular_intents}")
        
        # 3. 生成执行计划
        intent_plan = self._create_execution_plan(special_intents, regular_intents)
        
        # 4. 写入 state
        state["intent_plan"] = intent_plan
        state["route_decision"] = "intent_planner"
        
        print(f"[Planner] plan - 执行计划: mode={intent_plan['execution_mode']}, "
              f"branches={intent_plan['planned_branches']}")
        emit_trace("node_end", "intent_planner", f"规划完成: {intent_plan['execution_mode']}模式")
        
        return state
    
    def _normalize_intents(self, intents: list) -> list:
        """意图归一化 - 将别名转换为标准意图"""
        normalized = []
        for intent in intents:
            if not intent:
                continue
            # 归一化为小写
            intent_lower = intent.strip().lower()
            # 应用别名映射
            normalized_intent = INTENT_ALIASES.get(intent_lower, intent_lower)
            # 去重（避免归一化后重复）
            if normalized_intent not in normalized:
                normalized.append(normalized_intent)
        return normalized
    
    def _separate_intents(self, intents: list) -> tuple:
        """分离特殊意图和业务意图"""
        special = []
        regular = []
        
        for intent in intents:
            if intent in SPECIAL_INTENTS:
                special.append(intent)
            elif intent in BUSINESS_INTENTS:
                regular.append(intent)
            elif intent == "general":
                # general 作为降级选项，不加入 regular
                pass
            else:
                # 未知意图，忽略
                print(f"[Planner] _separate_intents - 忽略未知意图: {intent}")
        
        return special, regular
    
    def _create_execution_plan(self, special_intents: list, regular_intents: list) -> IntentPlan:
        """创建执行计划"""
        
        # 确定执行模式
        if len(regular_intents) == 0:
            execution_mode = "single"
        elif len(regular_intents) == 1:
            execution_mode = "single"
        else:
            execution_mode = "parallel"
        
        # 确定主意图
        primary_intent = regular_intents[0] if regular_intents else (special_intents[0] if special_intents else "general")
        
        # 确定次要意图
        secondary_intents = regular_intents[1:] if len(regular_intents) > 1 else []
        
        # 计划执行的分支
        planned_branches = []
        for intent in regular_intents:
            branch = INTENT_TO_BRANCH.get(intent)
            if branch and branch not in planned_branches:
                planned_branches.append(branch)
        
        # 确定是否需要特殊处理
        requires_special_handling = len(special_intents) > 0
        
        return IntentPlan(
            execution_mode=execution_mode,
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            special_intents=special_intents,
            regular_intents=regular_intents,
            planned_branches=planned_branches,
            requires_special_handling=requires_special_handling,
        )


# ============ 模块级实例 ============

_planner_instance = None


def get_planner() -> IntentPlanner:
    """获取全局规划器实例"""
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = IntentPlanner()
    return _planner_instance
