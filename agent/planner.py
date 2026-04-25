"""
意图规划层 - 三层解耦架构的核心

职责：
1. 意图归一化（food_report → food, workout_report → workout）
2. 特殊意图与业务意图分离
3. 生成执行计划（执行模式、分支列表）
4. 为每个分支生成专属的 prompt 上下文包（BranchPromptBundle）

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

import re
import traceback
from agent.state import AgentState, IntentPlan, DecomposedTask
from agent.stream_utils import emit_trace, emit_event

# ============ 常量定义 ============

# 特殊意图（会话控制层）
SPECIAL_INTENTS = {"confirm", "profile_update"}

# 业务意图（可并行执行）
BUSINESS_INTENTS = {"food", "food_report", "workout", "workout_report", "recipe", "stats_query", "recovery"}

# 意图 → 分支节点名映射
INTENT_TO_BRANCH = {
    "food": "food_branch",
    "food_report": "food_branch",
    "workout": "workout_advice_branch",
    "workout_report": "workout_report_branch",
    "recipe": "recipe_branch",
    "stats_query": "stats_branch",
    "recovery": "workout_advice_branch",  # recovery 复用 advice 分支（运动后拉伸/恢复指导）
}

# 子任务 intent 归一化映射（与 multi_agent._normalize_task_intent 保持一致）
_TASK_INTENT_NORMALIZE = {
    "food_report": "food",
    "recovery": "workout",
}


# ============ 子任务分解 ============

def decompose_tasks(input_message: str, intents: list) -> list:
    """将复合输入拆分为多个结构化子任务。

    策略：规则优先（按标点/连接词拆句 + 关键词匹配意图）。
    每个子任务包含 intent、branch、query 三个字段。

    Returns:
        DecomposedTask 列表；如果不需要分解或分解失败，返回 None。
    """
    print(f"[Decompose] decompose_tasks - 开始分解, intents={intents}")
    try:
        # 过滤掉非业务意图
        business_intents = [i for i in intents if i in BUSINESS_INTENTS]
        if not business_intents:
            print(f"[Decompose] decompose_tasks - 无业务意图，跳过分解")
            return None
        if len(business_intents) <= 1:
            print(f"[Decompose] decompose_tasks - 单业务意图，跳过分解")
            return None

        sub_sentences = _split_sub_sentences(input_message)
        print(f"[Decompose] decompose_tasks - 子句: {sub_sentences}")

        if len(sub_sentences) <= 1:
            print(f"[Decompose] decompose_tasks - 无法拆分输入，跳过分解")
            return None

        tasks = _match_clauses_to_intents(sub_sentences, business_intents)
        print(f"[Decompose] decompose_tasks - 匹配结果: {tasks}")

        if not tasks:
            return None

        # 如果只有一个 task 且内容等于原始输入，分解没有意义
        if len(tasks) == 1 and tasks[0]["query"] == input_message.strip():
            print(f"[Decompose] decompose_tasks - 单 task 等于原始输入，跳过")
            return None

        # 同一 branch 多个子任务合并文本
        tasks = _deduplicate_tasks(tasks)
        print(f"[Decompose] decompose_tasks - 最终结果: {tasks}")
        return tasks
    except Exception as e:
        print(f"[Decompose] decompose_tasks 分解失败: {e}")
        traceback.print_exc()
        return None


def _split_sub_sentences(text: str) -> list:
    """按标点和连接词拆句。

    分割优先级：句号/感叹号/问号/分号 > 连接词(另外/还有/同时/并且/然后) > 逗号
    """
    # 第一层：按句末标点拆分
    parts = re.split(r'[。！？!?；;\n]', text)
    sub_sentences = []
    for part in parts:
        # 第二层：按连接词拆分
        pieces = re.split(r'另外|还有|同时|并且|然后', part)
        for piece in pieces:
            # 第三层：按逗号拆分（中英文逗号、顿号）
            sub_parts = re.split(r'[，,、]', piece)
            for sub in sub_parts:
                clause = sub.strip()
                if clause:
                    sub_sentences.append(clause)
    return sub_sentences


def _match_clauses_to_intents(clauses: list, business_intents: list) -> list:
    """为每个子句匹配最可能的意图，生成 DecomposedTask 列表。"""
    intent_keywords = {
        "food_report": ["吃了", "喝了", "摄入了", "吃了一个", "吃了俩", "干掉", "干饭",
                        "早餐吃了", "午餐吃了", "晚餐吃了"],
        "workout_report": ["跑了", "做了运动", "健身了", "练了", "锻炼了", "运动了",
                           "跑步了", "游泳了", "骑车了", "走路了", "跳绳了", "打球了"],
        "food": ["热量", "卡路里", "营养", "蛋白质", "脂肪", "碳水"],
        "workout": ["怎么练", "如何练", "训练方法", "健身计划"],
        "recovery": ["拉伸", "放松", "恢复", "酸痛", "肌肉恢复", "运动后", "怎么拉伸",
                     "如何拉伸", "放松肌肉", "拉伸运动", "cool down", "stretching"],
        "recipe": ["食谱", "推荐", "怎么吃", "搭配", "菜单"],
        "stats_query": ["统计", "总量", "累计", "今日", "今天", "剩余", "消耗", "摄入"],
    }

    # 优先级（高优先级先匹配）
    priority = ["food_report", "workout_report", "recovery", "recipe", "stats_query", "food", "workout"]

    tasks = []
    for clause in clauses:
        matched_intent = None
        for intent in priority:
            if intent not in business_intents:
                continue
            keywords = intent_keywords.get(intent, [])
            if any(kw in clause for kw in keywords):
                matched_intent = intent
                break
        if not matched_intent:
            # 无法匹配的子句，跳过
            print(f"[Decompose] 子句无法匹配意图: {clause}")
            continue
        branch = INTENT_TO_BRANCH.get(matched_intent)
        if not branch:
            continue
        tasks.append(DecomposedTask(intent=matched_intent, branch=branch, query=clause))

    return tasks


def _deduplicate_tasks(tasks: list) -> list:
    """同一 branch 多个子任务合并文本（保留顺序）。"""
    branch_to_idx = {}
    deduplicated = []
    for task in tasks:
        branch = task["branch"]
        if branch in branch_to_idx:
            idx = branch_to_idx[branch]
            deduplicated[idx]["query"] += "，" + task["query"]
        else:
            branch_to_idx[branch] = len(deduplicated)
            deduplicated.append(task)
    return deduplicated


def decompose_tasks_node(state: AgentState) -> AgentState:
    """子任务分解节点 - 在 classify_intent 之后执行。

    根据 input_message + intents 生成结构化 decomposed_tasks。
    分解失败时安全降级（清空 decomposed_tasks，planner 回退到旧逻辑）。
    """
    print(f"[Decompose] decompose_tasks_node - 开始")
    emit_trace("node_start", "decompose_tasks", "正在分解子任务...")
    state["route_decision"] = "decompose_tasks"

    input_message = state.get("input_message", "")
    intents = state.get("intents", [])

    tasks = decompose_tasks(input_message, intents)
    state["decomposed_tasks"] = tasks

    if tasks:
        print(f"[Decompose] decompose_tasks_node - 分解完成: {len(tasks)} 个子任务")
        for i, task in enumerate(tasks):
            print(f"  task[{i}]: intent={task['intent']}, branch={task['branch']}, query={task['query'][:50]}...")
        emit_trace("decompose_tasks", "decompose_tasks", f"分解为 {len(tasks)} 个子任务")
    else:
        print(f"[Decompose] decompose_tasks_node - 未分解，planner 将回退到旧逻辑")
        emit_trace("decompose_tasks", "decompose_tasks", "未分解，使用旧逻辑")

    emit_trace("node_end", "decompose_tasks", "执行完成")
    return state


# ============ 意图规划器 ============

class IntentPlanner:
    """意图规划器 - 把意图集合转换为执行计划"""
    
    def __init__(self):
        pass
    
    def plan(self, state: AgentState) -> AgentState:
        """规划节点 - 在 classify_intent 之后执行，生成执行计划

        同时为每个分支生成专属的 prompt 上下文包（BranchPromptBundle），
        确保各分支只拿到自己需要的上下文，避免跨分支污染。

        优先从 decomposed_tasks 生成计划；没有则回退到旧的 intent-based 逻辑。
        """
        print(f"[Planner] plan - 开始意图规划")
        emit_trace("node_start", "intent_planner", "正在规划执行计划...")

        decomposed_tasks = state.get("decomposed_tasks")
        intents = state.get("intents", [state.get("intent", "general")])

        # ---- 优先：task-based 规划（仅当覆盖所有业务意图时）----
        if decomposed_tasks:
            # 覆盖校验：decomposed_tasks 必须覆盖所有业务意图对应的分支
            covered_branches = {t.get("branch") for t in decomposed_tasks if t.get("branch")}
            business_intents_in_input = [i for i in intents if i in BUSINESS_INTENTS]
            required_branches = {INTENT_TO_BRANCH[i] for i in business_intents_in_input if i in INTENT_TO_BRANCH}
            if required_branches and not required_branches.issubset(covered_branches):
                missing = required_branches - covered_branches
                print(f"[Planner] plan - decomposed_tasks 覆盖不全，缺少 {missing}，回退 intent-based")
                decomposed_tasks = None  # 降级到旧逻辑
                state["decomposed_tasks"] = None  # 同步清理 state，防止 fanout 读到失效数据

        if decomposed_tasks:
            print(f"[Planner] plan - 使用 decomposed_tasks ({len(decomposed_tasks)} 个)，覆盖校验通过")
            intent_plan = self._create_plan_from_tasks(decomposed_tasks, intents)
            branch_bundles = self._generate_task_based_bundles(decomposed_tasks, state)

            state["intent_plan"] = intent_plan
            state["branch_prompt_bundles"] = branch_bundles
            state["route_decision"] = "intent_planner"

            print(f"[Planner] plan - task-based 规划: mode={intent_plan['execution_mode']}, "
                  f"branches={intent_plan['planned_branches']}")
            print(f"[Planner] plan - 已生成 {len(branch_bundles)} 个分支 prompt bundle (task-based)")
            emit_trace("node_end", "intent_planner", f"规划完成: {intent_plan['execution_mode']}模式 (task-based)")
            return state

        # ---- 降级：intent-based 规划（旧逻辑）----
        print(f"[Planner] plan - 无 decomposed_tasks，使用 intent-based 旧逻辑")
        print(f"[Planner] plan - 输入意图: {intents}")

        # 1. 意图归一化
        normalized_intents = self._normalize_intents(intents)
        print(f"[Planner] plan - 归一化后: {normalized_intents}")

        # 2. 分离特殊意图和业务意图
        special_intents, regular_intents = self._separate_intents(normalized_intents)
        print(f"[Planner] plan - 特殊意图: {special_intents}, 业务意图: {regular_intents}")

        # 3. 生成执行计划
        intent_plan = self._create_execution_plan(special_intents, regular_intents)

        # 4. 为每个分支生成 prompt bundle
        branch_bundles = self._generate_branch_bundles(intent_plan, state)

        # 5. 写入 state
        state["intent_plan"] = intent_plan
        state["branch_prompt_bundles"] = branch_bundles
        state["route_decision"] = "intent_planner"

        print(f"[Planner] plan - 执行计划: mode={intent_plan['execution_mode']}, "
              f"branches={intent_plan['planned_branches']}")
        print(f"[Planner] plan - 已生成 {len(branch_bundles)} 个分支 prompt bundle")
        emit_trace("node_end", "intent_planner", f"规划完成: {intent_plan['execution_mode']}模式")

        return state
    
    def _normalize_intents(self, intents: list) -> list:
        """意图归一化 - 小写化 + 去重，不做别名合并

        注意：不做 food_report→food / workout_report→workout 的别名合并，
        因为 report 语义（主动报告→自动确认）需要通过 source_intents 传递给子 agent。
        分支路由由 INTENT_TO_BRANCH 直接映射，不需要先归一化。
        """
        normalized = []
        for intent in intents:
            if not intent:
                continue
            intent_lower = intent.strip().lower()
            if intent_lower not in normalized:
                normalized.append(intent_lower)
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
    
    def _generate_branch_bundles(self, intent_plan: IntentPlan, state: AgentState) -> dict:
        """为每个分支生成专属的 prompt 上下文包

        目标：
        1. 每个分支只拿到自己需要的上下文
        2. 共享上下文（用户档案、今日统计）统一格式化
        3. 分支上下文（如 recipe 的剩余热量）独立构造
        4. 避免跨分支上下文污染

        Returns:
            dict: 分支名 → BranchPromptBundle 映射
        """
        from agent.context_manager import build_branch_prompt_bundle

        bundles = {}
        planned_branches = intent_plan.get("planned_branches", [])
        regular_intents = intent_plan.get("regular_intents", [])

        # 构建分支名 → 意图的映射
        branch_to_intent = {}
        for intent in regular_intents:
            branch = INTENT_TO_BRANCH.get(intent)
            if branch and branch not in branch_to_intent:
                branch_to_intent[branch] = intent

        # 为每个分支生成 bundle
        for branch_name in planned_branches:
            intent = branch_to_intent.get(branch_name, "general")
            try:
                bundle = build_branch_prompt_bundle(branch_name, intent, state)
                bundles[branch_name] = bundle
                print(f"[Planner] _generate_branch_bundles - 生成 {branch_name} bundle (intent={intent})")
            except Exception as e:
                print(f"[Planner] _generate_branch_bundles - 生成 {branch_name} bundle 失败: {e}")
                # 使用最小化 bundle 作为 fallback
                bundles[branch_name] = {
                    "branch_name": branch_name,
                    "intent": intent,
                    "branch_input": state.get("input_message", ""),
                    "system_context": "",
                    "shared_context": "",
                    "branch_context": "",
                    "extra_sections": {},
                    "conversation_window": [],
                }

        return bundles

    def _create_plan_from_tasks(self, decomposed_tasks: list, intents: list) -> IntentPlan:
        """从 decomposed_tasks 创建执行计划。

        注意：special_intents（confirm/profile_update）必须从原始 intents 提取，
        因为 decomposed_tasks 只包含业务意图，不包含特殊意图。
        """
        planned_branches = []
        task_intents = []
        for task in decomposed_tasks:
            branch = task.get("branch")
            if branch and branch not in planned_branches:
                planned_branches.append(branch)
            intent = task.get("intent")
            if intent and intent not in task_intents:
                task_intents.append(intent)

        # 从原始 intents 提取特殊意图（decomposed_tasks 不含特殊意图）
        all_normalized = self._normalize_intents(intents)
        special_intents, _ = self._separate_intents(all_normalized)

        # 业务意图从 task 提取并归一化
        normalized_task = self._normalize_intents(task_intents)
        _, regular_intents = self._separate_intents(normalized_task)

        execution_mode = "single" if len(planned_branches) <= 1 else "parallel"
        primary_intent = regular_intents[0] if regular_intents else "general"
        secondary_intents = regular_intents[1:] if len(regular_intents) > 1 else []

        return IntentPlan(
            execution_mode=execution_mode,
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            special_intents=special_intents,
            regular_intents=regular_intents,
            planned_branches=planned_branches,
            requires_special_handling=len(special_intents) > 0,
        )

    def _generate_task_based_bundles(self, decomposed_tasks: list, state: AgentState) -> dict:
        """从 decomposed_tasks 为每个分支生成 prompt bundle。

        与 _generate_branch_bundles 的关键区别：
        - branch_input 优先取 task.query（而非从原始 input_message 抽取）
        - 每个 task 已经有明确的 intent 和 branch 映射
        """
        from agent.context_manager import build_branch_prompt_bundle

        # 构建 branch → task 映射（同一 branch 取第一个）
        branch_to_task = {}
        for task in decomposed_tasks:
            branch = task.get("branch")
            if branch and branch not in branch_to_task:
                branch_to_task[branch] = task

        bundles = {}
        for branch_name, task in branch_to_task.items():
            intent = task.get("intent", "general")
            normalized_intent = _TASK_INTENT_NORMALIZE.get(intent, intent)
            try:
                bundle = build_branch_prompt_bundle(branch_name, normalized_intent, state)
                # 覆盖 branch_input：优先使用 task.query
                bundle["branch_input"] = task.get("query", state.get("input_message", ""))
                bundle["intent"] = intent  # 保留原始 intent
                bundles[branch_name] = bundle
                print(f"[Planner] _generate_task_based_bundles - 生成 {branch_name} bundle "
                      f"(intent={intent}, query={task.get('query', '')[:30]}...)")
            except Exception as e:
                print(f"[Planner] _generate_task_based_bundles - 生成 {branch_name} bundle 失败: {e}")
                bundles[branch_name] = {
                    "branch_name": branch_name,
                    "intent": intent,
                    "branch_input": task.get("query", state.get("input_message", "")),
                    "system_context": "",
                    "shared_context": "",
                    "branch_context": "",
                    "extra_sections": {},
                    "conversation_window": [],
                }

        return bundles


# ============ 模块级实例 ============

_planner_instance = None


def get_planner() -> IntentPlanner:
    """获取全局规划器实例"""
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = IntentPlanner()
    return _planner_instance
