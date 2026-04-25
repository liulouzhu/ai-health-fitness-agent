"""
统一上下文管理模块

职责：
- 提供 ContextBundle 数据结构，分层组织运行时上下文
- 按 intent 策略选择性加载不同类别的上下文，避免一股脑塞入
- Token 预算管理：每层独立裁剪，总量保护
- 收敛各 agent 的上下文拼装逻辑，统一入口

数据流：
  memory_agent (持久化)
       ↓
  ContextManager.build_context() / build_prompt_messages()
       ↓
  各 Agent（只消费结构化上下文，不直接读 memory_agent）
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from agent.state import AgentState
from agent.memory import get_memory_agent
from config import AgentConfig

logger = logging.getLogger(__name__)

# ============ 共享工具函数 ============


def is_retrieval_sufficient(
    retrieved_content: str, query: str = None, domain: str = "通用"
) -> bool:
    """
    判断检索内容是否足够。

    Args:
        retrieved_content: 检索返回的文本内容
        query: 原始用户问题（可选，部分场景需要）
        domain: 领域标识，用于构造合适的判断提示词
    """
    if not retrieved_content:
        return False

    from agent.llm import get_llm

    llm = get_llm()

    if domain == "recipe":
        judge_prompt = (
            f"判断以下检索内容是否足够推荐食谱。\n\n检索内容：{retrieved_content[:2000]}\n\n"
            f'如果检索内容足够，返回"足够"。如果不足，返回"不足"。\n只返回"足够"或"不足"。'
        )
    elif domain == "workout":
        judge_prompt = (
            f"判断以下检索内容是否足够回答用户的问题。\n\n"
            f"用户问题：{query}\n\n检索内容：{retrieved_content[:2000]}\n\n"
            f'如果检索内容足够回答问题，返回"足够"。如果不足，返回"不足"。\n只返回"足够"或"不足"，不要其他文字。'
        )
    else:
        judge_prompt = (
            f"判断以下检索内容是否足够回答用户的问题。\n\n检索内容：{retrieved_content[:2000]}\n\n"
            f'如果检索内容足够，返回"足够"。如果不足，返回"不足"。\n只返回"足够"或"不足"。'
        )

    response = llm.invoke([{"role": "user", "content": judge_prompt}])
    return "足够" in response.content


# ============ 分层数据结构 ============


class ContextBundle(TypedDict, total=False):
    """统一上下文数据结构，所有字段均按语义分层"""

    system_context: str  # system prompt 文本
    extra_context: str  # intent-specific 额外上下文片段
    conversation_window: List[Dict]  # 运行时滑动窗口消息
    user_memory: Dict  # profile + preferences (raw dict)
    task_context: Dict  # intent-specific 业务上下文
    retrieved_knowledge: str  # 外部检索内容


# ============ Token 估算器 ============


class TokenEstimator:
    """
    文本 token 数估算器

    优先使用 tiktoken 真实 encoder，失败时自动降级为
    字符数 / CHARS_PER_TOKEN_ESTIMATE 近似估算。
    """

    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self._encoder = None
        self._using_fallback = False
        self._init_encoder()

    def _init_encoder(self) -> None:
        """初始化 encoder，失败时降级为 fallback"""
        if self.mode == "chars":
            self._using_fallback = True
            return

        # 尝试加载 tiktoken（langchain-openai 自带）
        try:
            import tiktoken

            self._encoder = tiktoken.get_encoding("cl100k_base")
            self._using_fallback = False
            logger.debug("[TokenEstimator] Using tiktoken (cl100k_base)")
        except Exception as e:
            logger.debug(
                f"[TokenEstimator] tiktoken unavailable ({e}), using char-based fallback"
            )
            self._using_fallback = True

    def estimate(self, text: str) -> int:
        """
        估算文本的 token 数

        Returns:
            int: 估算 token 数
        """
        if not text:
            return 0
        if self._using_fallback:
            # 字符数 / 3.5 是中英文混合文本的合理近似
            return max(1, int(len(text) / AgentConfig.CHARS_PER_TOKEN_ESTIMATE))
        try:
            return len(self._encoder.encode(text))
        except Exception:
            return max(1, int(len(text) / AgentConfig.CHARS_PER_TOKEN_ESTIMATE))

    def estimate_messages(self, messages: List[Dict]) -> int:
        """估算一组消息的 total token 数（含 role/content overhead）"""
        total = 0
        for msg in messages:
            # 每个消息有 ~4 token overhead（role 标签 + 格式）
            total += 4
            total += self.estimate(msg.get("content", ""))
        return total

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback


# 全局 TokenEstimator 实例
_token_estimator: Optional[TokenEstimator] = None


def get_token_estimator() -> TokenEstimator:
    global _token_estimator
    if _token_estimator is None:
        _token_estimator = TokenEstimator(mode=AgentConfig.TOKEN_ESTIMATOR_MODE)
    return _token_estimator


# ============ 工具函数 ============


def _truncate_to_tokens(text: str, max_tokens: int, estimator: TokenEstimator) -> str:
    """按 token 数上限截断文本（保留开头）"""
    if max_tokens <= 0:
        return ""
    tokens = estimator.estimate(text)
    if tokens <= max_tokens:
        return text

    # 二分查找最大字符数
    lo, hi = 0, len(text)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if estimator.estimate(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid
    return text[:lo]


def _format_pending_stats(pending: dict) -> str:
    """将待确认数据格式化为易读的描述文本"""
    if not pending:
        return "（无待确认记录）"

    parts = []
    ptype = pending.get("type", "unknown")

    if ptype == "multi":
        if pending.get("food"):
            food = pending["food"]
            parts.append(
                f"食物记录：{food.get('name', '?')}（{food.get('calories', 0)} kcal，{food.get('protein', 0)}g 蛋白）"
            )
        if pending.get("workout"):
            workout = pending["workout"]
            parts.append(
                f"运动记录：{workout.get('type', '?')}（{workout.get('duration', 0)}分钟，{workout.get('calories', 0)} kcal）"
            )
    elif ptype == "meal":
        data = pending.get("data", {})
        parts.append(
            f"食物记录：{data.get('name', '?')}（{data.get('calories', 0)} kcal，{data.get('protein', 0)}g 蛋白）"
        )
    elif ptype == "workout":
        data = pending.get("workout", pending.get("data", {}))
        parts.append(
            f"运动记录：{data.get('type', '?')}（{data.get('duration', 0)}分钟，{data.get('calories', 0)} kcal）"
        )

    if not parts:
        return "（待确认数据格式未知）"

    return "、".join(parts)


# ============ 系统人设提示词 ============

SYSTEM_PROMPTS: Dict[str, str] = {
    "food": """你是一个食物营养分析专家。请分析用户询问的食物并提供营养信息。

分析内容：
- 食物名称
- 热量 (kcal)
- 蛋白质 (g)
- 脂肪 (g)
- 碳水化合物 (g)

直接回复分析结果，不需要额外解释。""",
    "workout": """你是一个健身教练专家。请根据以下信息回答用户的问题。

规则：
- 如果用户同时包含运动记录和运动咨询，你必须同时处理：先确认运动记录，再回答运动问题
- 回答运动问题时，给出具体动作名称、组数、次数、注意事项等专业建议
- 直接回复内容，不需要额外解释""",
    "recipe": """你是一个营养师。根据用户的饮食目标和限制，推荐合适的食谱。

请根据以上信息，推荐合适的食谱组合，确保：
1. 总热量不超过剩余热量
2. 蛋白质尽量达到目标
3. 食物种类多样化
4. **严格避免推荐用户不喜欢或过敏的食物**

直接回复推荐内容，不需要额外解释。""",
    "general": """你是一个健身健康智能助手，可以进行日常对话。
请回复用户，保持对话连贯性。如果用户问到健身或饮食相关问题，可以适当引导。""",
    "classify_intent": """你是一个意图分类器。你的唯一任务是识别用户输入的意图类别。

规则：
1. 只返回意图标签，不回答用户的问题
2. 不要提供任何解释、建议或内容
3. 如果用户问问题，只判断意图，不回答问题
4. 重要：用户一句话可能包含多个意图，必须全部识别出来！

可选意图标签（只选这些中的一个或多个）：
- food: 用户想分析食物营养或询问食物信息
- food_report: 用户主动报告吃了什么（如"我吃了..."、"吃了..."）
- workout: 用户想了解运动或健身信息（如"怎么练..."、"如何拉伸"）
- workout_report: 用户主动报告做了什么运动（如"跑了..."、"练了..."）
- recipe: 用户想要食谱推荐
- stats_query: 用户想查看统计数据
- profile_update: 用户想更新个人档案
- confirm: 用户在确认或取消
- general: 通用对话或其他意图

示例：
- "我吃了鸡腿" → food_report
- "跑了10公里" → workout_report
- "我吃了鸡腿，跑了10公里" → food_report,workout_report
- "我吃了鸡腿，应该如何拉伸？" → food_report,workout
- "我吃了鸡腿，跑了10公里，如何拉伸？" → food_report,workout_report,workout

输出格式：只返回意图标签，多个用逗号分隔。例如：food,workout

重要：你是一个分类器，不是助手。不要回答问题，只分类意图。
重要：必须检查用户输入的每一部分，不要遗漏任何意图！""",
}


# ============ 上下文管理器 ============


class ContextManager:
    """
    统一上下文装配器 + Token 预算管理

    公共 API：
        build_context(intent, state, retrieved_content)
        build_prompt_messages(intent, state, retrieved_content, extra_sections, user_input)
        append_messages(state, user_msg, assistant_msg)
        get_preferences_str()         ← 公开偏好格式化
        format_task_context(intent)    ← 公开 task_context 格式化
        estimate_tokens(text)         ← 公开 token 估算
    """

    def __init__(self):
        self._memory = None
        self._estimator = None

    @property
    def memory(self):
        if self._memory is None:
            self._memory = get_memory_agent()
        return self._memory

    @property
    def estimator(self) -> TokenEstimator:
        if self._estimator is None:
            self._estimator = get_token_estimator()
        return self._estimator

    # ---- 公共 API ----

    def build_context(
        self,
        intent: str,
        state: AgentState,
        retrieved_content: Optional[str] = None,
    ) -> ContextBundle:
        """
        按 intent 策略构建分层上下文（不含裁剪）

        Returns:
            ContextBundle 分层字典
        """
        intent = intent or "general"
        system_context = SYSTEM_PROMPTS.get(intent, SYSTEM_PROMPTS["general"])
        
        # 意图分类：不需要上下文、记忆、任务上下文，只用当前输入
        if intent == "classify_intent":
            return ContextBundle(
                system_context=system_context,
                extra_context="",
                conversation_window=[],  # 不使用对话历史
                user_memory={},  # 不使用用户记忆
                task_context={},  # 不使用任务上下文
                retrieved_knowledge="",
            )
        
        conversation_window = self._get_conversation_window(state)
        user_memory = self._get_user_memory()
        task_context = self._build_task_context(intent, state)
        extra_context = self._build_extra_context(intent, task_context)
        retrieved_knowledge = (retrieved_content or "").strip()

        return ContextBundle(
            system_context=system_context,
            extra_context=extra_context,
            conversation_window=conversation_window,
            user_memory=user_memory,
            task_context=task_context,
            retrieved_knowledge=retrieved_knowledge,
        )

    def build_prompt_messages(
        self,
        intent: str,
        state: AgentState,
        retrieved_content: Optional[str] = None,
        extra_sections: Optional[Dict[str, str]] = None,
        user_input: Optional[str] = None,
    ) -> List[Dict]:
        """
        构建可直接发送给 LLM 的消息列表（含 token 预算裁剪）

        Args:
            intent: 当前意图
            state: LangGraph AgentState
            retrieved_content: 外部检索内容
            extra_sections: intent-specific 动态片段，key 为段落标题，value 为内容文本
                            例：{"用户问题": "...", "参考内容": "..."}
            user_input: 显式传入的用户输入（默认从 state 取）
        """
        est = self.estimator
        cfg = AgentConfig

        bundle = self.build_context(intent, state, retrieved_content)
        user_input = user_input or state.get("input_message", "")

        # ----- Section 1: system_context（固定文本，不裁剪） -----
        system_tokens = est.estimate(bundle["system_context"])
        sections = {
            "system_context": bundle["system_context"],
        }

        # ----- Section 2: extra_context（token 预算裁剪） -----
        extra_text = bundle["extra_context"]
        if extra_text:
            extra_tokens = est.estimate(extra_text)
            if extra_tokens > cfg.MAX_EXTRA_CONTEXT_TOKENS:
                extra_text = _truncate_to_tokens(
                    extra_text, cfg.MAX_EXTRA_CONTEXT_TOKENS, est
                )
                logger.debug(
                    f"[CtxMgr] extra_context trimmed: {extra_tokens} → {est.estimate(extra_text)} tokens"
                )
            sections["extra_context"] = extra_text

        # ----- Section 3: extra_sections（agent 动态内容，token 预算裁剪） -----
        if extra_sections:
            for key, value in extra_sections.items():
                if not value:
                    continue
                max_tok = cfg.MAX_TASK_CONTEXT_TOKENS
                value_tokens = est.estimate(value)
                if value_tokens > max_tok:
                    value = _truncate_to_tokens(value, max_tok, est)
                    logger.debug(
                        f"[CtxMgr] extra_section '{key}' trimmed: {value_tokens} → {est.estimate(value)} tokens"
                    )
                sections[key] = value

        # ----- Section 4: conversation_window（token 预算裁剪） -----
        conv = bundle["conversation_window"]
        # 估算已有 sections 消耗
        consumed = (
            sum(est.estimate(v) for v in sections.values()) + 4
        )  # +4 = user msg overhead
        remaining = cfg.MAX_CONVERSATION_WINDOW_TOKENS
        trimmed_conv = []
        for msg in reversed(conv):
            t = est.estimate(msg.get("content", ""))
            if remaining >= t + 4:
                trimmed_conv.insert(0, msg)
                remaining -= t + 4
            else:
                break
        sections["conversation_window"] = trimmed_conv

        # ----- 汇总 token 报告 -----
        total = est.estimate_messages(
            [
                {"role": "system", "content": v} if k != "conversation_window" else v
                for k, v in sections.items()
                if k != "conversation_window"
            ]
            + list(sections.get("conversation_window", []))
        )
        logger.debug(
            f"[CtxMgr] intent={intent} | "
            f"system={system_tokens} | extra={est.estimate(sections.get('extra_context', ''))} | "
            f"conv={est.estimate_messages(sections.get('conversation_window', []))} | "
            f"total≈{total} tokens (budget={cfg.MAX_TOTAL_CONTEXT_TOKENS})"
        )

        # ----- 构建最终消息列表 -----
        messages = [{"role": "system", "content": sections["system_context"]}]

        extra_all = []
        if sections.get("extra_context"):
            extra_all.append(sections["extra_context"])
        if extra_sections:
            for key, value in sections.items():
                if (
                    key
                    not in ("system_context", "extra_context", "conversation_window")
                    and value
                ):
                    extra_all.append(f"{key}：{value}")

        if extra_all:
            messages.append({"role": "system", "content": "\n".join(extra_all)})

        for msg in sections.get("conversation_window", []):
            messages.append(msg)

        messages.append({"role": "user", "content": user_input})
        return messages

    def append_messages(
        self, state: AgentState, user_msg: str, assistant_msg: str
    ) -> None:
        """
        向 state['messages'] 追加一轮对话，并维持滑动窗口上限。
        
        注意：此方法直接修改 state["messages"]，每轮只应调用一次。
        多意图场景下，由主节点（如 general_node）调用，分支节点不应调用。
        """
        messages = state.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        
        # 追加新消息
        messages = messages + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        
        # 强制窗口限制
        if len(messages) > AgentConfig.MAX_RECENT_MESSAGES:
            messages = messages[-AgentConfig.MAX_RECENT_MESSAGES:]
        
        state["messages"] = messages
        print(f"[ContextManager] append_messages - 追加后 messages 数量: {len(messages)}")

    # ---- 公开工具方法 ----

    def get_preferences_str(self) -> str:
        """
        公开 API：将 preferences 字典格式化为上下文字符串。
        """
        return self._get_preferences_str()

    def format_task_context(
        self, intent: str, task_context: Optional[Dict] = None
    ) -> str:
        """
        公开 API：将 task_context 格式化为易读文本，供 agent 注入 prompt。

        Args:
            intent: 当前意图
            task_context: 可选，显式传入（否则从 memory 加载）
        """
        if task_context is None:
            task_context = self._build_task_context(intent, {})
        return self._format_task_context_text(intent, task_context)

    def estimate_tokens(self, text: str) -> int:
        """公开 API：估算文本 token 数"""
        return self.estimator.estimate(text)

    # ---- 内部方法 ----

    def _get_conversation_window(self, state: AgentState) -> List[Dict]:
        """从 state['messages'] 提取运行时滑动窗口"""
        messages = state.get("messages", [])
        return messages[-AgentConfig.MAX_RECENT_MESSAGES :]
    
    def _get_classify_conversation_window(self, state: AgentState) -> List[Dict]:
        """为意图分类提取简短的对话窗口
        
        意图分类任务不需要完整的对话历史，只使用当前用户输入。
        如果需要上下文，只使用最近 1 轮的用户消息（不包括 assistant 回复），
        避免 assistant 的业务回答污染分类结果。
        """
        messages = state.get("messages", [])
        if not messages:
            return []
        
        # 只取最近 2 条用户消息（当前 + 上一轮），不包括 assistant 回复
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        recent_users = user_messages[-2:] if len(user_messages) >= 2 else user_messages
        
        # 截断过长的用户消息
        filtered = []
        for msg in recent_users:
            content = msg.get("content", "")
            # 用户消息截断到 100 字符
            filtered.append({
                "role": "user",
                "content": content[:100] + "..." if len(content) > 100 else content
            })
        
        return filtered

    def _get_user_memory(self) -> Dict:
        """加载用户长期记忆：profile + preferences"""
        profile = self.memory.load_profile()
        preferences = self.memory.load_preferences()
        return {"profile": profile, "preferences": preferences}

    def _build_task_context(self, intent: str, state: AgentState) -> Dict:
        """按 intent 加载不同的业务上下文"""
        if intent in ("food", "food_report"):
            return self._build_food_context()
        elif intent in ("workout", "workout_report"):
            return self._build_workout_context()
        elif intent == "recipe":
            return self._build_recipe_context()
        elif intent == "stats_query":
            return self._build_stats_context()
        elif intent == "confirm":
            # 确认意图的上下文数据由 graph state 中的 pending_confirmation 提供，
            # 不再从 pending_stats.json 读取（该文件仅作旧版兼容 fallback）
            return {}
        return {}

    def _build_food_context(self) -> Dict:
        today = self.memory.load_daily_stats()
        profile = self.memory.load_profile()
        return {"daily_stats": today, "profile": profile}

    def _build_workout_context(self) -> Dict:
        prefs = self.memory.load_preferences()
        today = self.memory.load_daily_stats()
        return {
            "workout_preferences": prefs.get("workout_preferences", {}),
            "daily_stats": today,
        }

    def _build_recipe_context(self) -> Dict:
        profile = self.memory.load_profile()
        today = self.memory.load_daily_stats()
        prefs = self.memory.load_preferences()
        target_cal = int(
            profile.get("target_calories", AgentConfig.DEFAULT_TARGET_CALORIES)
        )
        target_pro = int(
            profile.get("target_protein", AgentConfig.DEFAULT_TARGET_PROTEIN)
        )
        consumed_cal = today.get("consumed_calories", 0)
        consumed_pro = today.get("consumed_protein", 0)
        burned_cal = today.get("burned_calories", 0)
        return {
            "target_calories": target_cal,
            "target_protein": target_pro,
            "consumed_calories": consumed_cal,
            "consumed_protein": consumed_pro,
            "burned_calories": burned_cal,
            "remaining_calories": max(0, target_cal - consumed_cal + burned_cal),
            "remaining_protein": max(0, target_pro - consumed_pro),
            "goal": profile.get("goal", "维持"),
            "preferences": prefs,
            "daily_stats": today,
        }

    def _build_stats_context(self) -> Dict:
        today = self.memory.load_daily_stats()
        profile = self.memory.load_profile()
        return {"daily_stats": today, "profile": profile}

    def _build_extra_context(self, intent: str, task_context: Dict) -> str:
        """
        在 system prompt 之后、history 之前，注入意图特定的额外上下文片段。
        返回空字符串表示不需要额外注入。
        """
        parts = []
        prefs_str = self._get_preferences_str()

        if intent == "general":
            # 只在 general 意图下注入长期记忆（传入 intent 用于加权选择）
            longterm = self.memory.get_longterm_memory_context(
                limit=AgentConfig.MAX_LONGTERM_MEMORY_ITEMS,
                current_intent=intent,
            )
            if prefs_str:
                parts.append(f"用户偏好：{prefs_str}")
            if longterm:
                parts.append(f"长期记忆：{longterm}")

        elif intent in ("food", "food_report"):
            stats = task_context.get("daily_stats", {})
            if stats:
                parts.append(self._fmt_daily_intake(stats))

        elif intent in ("workout", "workout_report"):
            parts.extend(
                self._fmt_workout_prefs(task_context.get("workout_preferences", {}))
            )

        elif intent == "confirm":
            # 注意：确认流程的真实数据来自 graph state 的 pending_confirmation，
            # 不再依赖 pending_stats.json。如果 task_context 中有 pending_confirmation
            # 的相关信息（由调用方注入），会由其他分支处理。
            # pending_stats.json 作为旧版兼容 fallback，不作为主流程数据源。
            pass

        return "\n".join(parts)

    def _format_task_context_text(self, intent: str, task_context: Dict) -> str:
        """将 task_context 格式化为易读文本（用于 agent 的 extra_sections）"""
        parts = []

        if intent in ("food", "food_report"):
            stats = task_context.get("daily_stats", {})
            if stats:
                parts.append(self._fmt_daily_intake(stats))

        elif intent in ("workout", "workout_report"):
            parts.extend(
                self._fmt_workout_prefs(task_context.get("workout_preferences", {}))
            )
            stats = task_context.get("daily_stats", {})
            if stats and stats.get("burned_calories"):
                parts.append(f"今日已消耗：{int(stats.get('burned_calories', 0))} kcal")

        elif intent == "recipe":
            parts.append(f"剩余热量：{task_context.get('remaining_calories', 0)} kcal")
            parts.append(f"剩余蛋白质：{task_context.get('remaining_protein', 0)} g")
            parts.append(f"健身目标：{task_context.get('goal', '维持')}")

        elif intent == "stats_query":
            stats = task_context.get("daily_stats", {})
            profile = task_context.get("profile", {})
            if stats:
                target_cal = int(
                    profile.get("target_calories", AgentConfig.DEFAULT_TARGET_CALORIES)
                )
                target_pro = int(
                    profile.get("target_protein", AgentConfig.DEFAULT_TARGET_PROTEIN)
                )
                remaining = max(
                    0,
                    target_cal
                    - stats.get("consumed_calories", 0)
                    + stats.get("burned_calories", 0),
                )
                parts.append(
                    f"今日摄入：{int(stats.get('consumed_calories', 0))} / {target_cal} kcal"
                )
                parts.append(
                    f"今日蛋白：{stats.get('consumed_protein', 0):.0f} / {target_pro} g"
                )
                parts.append(f"今日消耗：{stats.get('burned_calories', 0)} kcal")
                parts.append(f"剩余热量额度：{remaining} kcal")

        return "\n".join(parts) if parts else ""

    # ---- 格式化辅助方法（消除重复） ----

    def _fmt_daily_intake(self, stats: Dict) -> str:
        """格式化"今日已摄入"片段"""
        return (
            f"今日已摄入：{int(stats.get('consumed_calories', 0))} kcal，"
            f"{stats.get('consumed_protein', 0):.0f}g 蛋白"
        )

    def _fmt_workout_prefs(self, workout_prefs: Dict) -> List[str]:
        """格式化运动偏好片段列表"""
        parts = []
        if workout_prefs.get("limitations"):
            parts.append(f"运动限制：{', '.join(workout_prefs['limitations'])}")
        if workout_prefs.get("disliked"):
            parts.append(f"不喜欢运动：{', '.join(workout_prefs['disliked'])}")
        return parts

    def _get_preferences_str(self) -> str:
        """将 preferences 字典格式化为上下文字符串（委托给 memory_agent）"""
        return self.memory.get_preferences_for_context()


# ============ 分支 Prompt Bundle 构建 ============


def build_branch_prompt_bundle(
    branch_name: str,
    intent: str,
    state: AgentState,
) -> dict:
    """
    为指定分支构建 prompt 上下文包。

    目标：
    - 每个分支只拿到自己需要的上下文
    - 共享上下文（用户档案、今日统计）统一格式化
    - 分支上下文（如 recipe 的剩余热量）独立构造
    - 避免跨分支上下文污染

    Args:
        branch_name: 分支名（food_branch / workout_branch / stats_branch / recipe_branch）
        intent: 该分支对应的意图（归一化后）
        state: LangGraph AgentState

    Returns:
        BranchPromptBundle 字典
    """
    from agent.state import BranchPromptBundle
    ctx_mgr = get_context_manager()

    # 1. System context
    system_context = SYSTEM_PROMPTS.get(intent, SYSTEM_PROMPTS.get("general", ""))

    # 2. Shared context（所有分支共享）
    shared_parts = []
    profile = {}
    daily_stats = {}
    
    try:
        profile = ctx_mgr.memory.load_profile() or {}
    except Exception as e:
        logger.debug(f"[build_branch_prompt_bundle] load_profile failed: {e}")
    
    try:
        daily_stats = ctx_mgr.memory.load_daily_stats() or {}
    except Exception as e:
        logger.debug(f"[build_branch_prompt_bundle] load_daily_stats failed: {e}")
    
    if profile:
        goal = profile.get("goal", "维持")
        shared_parts.append(f"健身目标：{goal}")
    if daily_stats:
        shared_parts.append(
            f"今日已摄入：{int(daily_stats.get('consumed_calories', 0))} kcal，"
            f"{daily_stats.get('consumed_protein', 0):.0f}g 蛋白"
        )
    shared_context = "\n".join(shared_parts)

    # 3. Branch-specific context
    branch_context = ""
    extra_sections = {}

    if intent in ("food", "food_report"):
        # Food branch: 饮食偏好、今日摄入
        prefs = ""
        try:
            prefs = ctx_mgr.get_preferences_str() or ""
        except Exception:
            pass
        if prefs:
            branch_context = f"用户偏好：{prefs}"
        extra_sections["用户偏好"] = prefs or "（暂无偏好记录）"
        
        if daily_stats:
            extra_sections["今日情况"] = (
                f"今日已摄入：{int(daily_stats.get('consumed_calories', 0))} kcal，"
                f"{daily_stats.get('consumed_protein', 0):.0f}g 蛋白"
            )

    elif intent in ("workout", "workout_report", "recovery"):
        # Workout branch: 运动偏好、今日消耗
        workout_prefs = {}
        try:
            prefs_data = ctx_mgr.memory.load_preferences() or {}
            workout_prefs = prefs_data.get("workout_preferences", {})
        except Exception:
            pass
        
        parts = []
        if workout_prefs.get("limitations"):
            parts.append(f"运动限制：{', '.join(workout_prefs['limitations'])}")
        if workout_prefs.get("disliked"):
            parts.append(f"不喜欢运动：{', '.join(workout_prefs['disliked'])}")
        if parts:
            branch_context = "\n".join(parts)
        
        if daily_stats and daily_stats.get("burned_calories"):
            extra_sections["今日消耗"] = f"今日已消耗：{int(daily_stats.get('burned_calories', 0))} kcal"
        
        prefs = ""
        try:
            prefs = ctx_mgr.get_preferences_str() or ""
        except Exception:
            pass
        extra_sections["用户偏好"] = prefs or "（暂无偏好记录）"

    elif intent == "stats_query":
        # Stats branch: 今日统计、目标对比
        if daily_stats and profile:
            target_cal = int(profile.get("target_calories", 2000))
            target_pro = int(profile.get("target_protein", 120))
            consumed_cal = daily_stats.get("consumed_calories", 0)
            consumed_pro = daily_stats.get("consumed_protein", 0)
            burned_cal = daily_stats.get("burned_calories", 0)
            remaining = max(0, target_cal - consumed_cal + burned_cal)
            branch_context = (
                f"目标热量：{target_cal} kcal，已摄入：{consumed_cal} kcal，剩余：{remaining} kcal\n"
                f"目标蛋白：{target_pro}g，已摄入：{consumed_pro:.0f}g\n"
                f"今日消耗：{burned_cal} kcal"
            )

    elif intent == "recipe":
        # Recipe branch: 剩余热量/蛋白、饮食目标
        if daily_stats and profile:
            target_cal = int(profile.get("target_calories", 2000))
            target_pro = int(profile.get("target_protein", 120))
            consumed_cal = daily_stats.get("consumed_calories", 0)
            consumed_pro = daily_stats.get("consumed_protein", 0)
            burned_cal = daily_stats.get("burned_calories", 0)
            remaining_cal = max(0, target_cal - consumed_cal + burned_cal)
            remaining_pro = max(0, target_pro - consumed_pro)
            goal = profile.get("goal", "维持")
            branch_context = (
                f"剩余热量：{remaining_cal} kcal\n"
                f"剩余蛋白质：{remaining_pro} g\n"
                f"健身目标：{goal}"
            )
            extra_sections["用户营养约束"] = (
                f"剩余热量：{remaining_cal} kcal；"
                f"剩余蛋白质：{remaining_pro} g；"
                f"健身目标：{goal}"
            )
            prefs = ""
            try:
                prefs = ctx_mgr.get_preferences_str() or ""
            except Exception:
                pass
            extra_sections["用户偏好"] = prefs or "（暂无偏好记录）"

    # 4. Conversation window（可以按分支裁剪）
    conv_window = []
    try:
        conv_window = ctx_mgr._get_conversation_window(state)
    except Exception:
        pass

    # 5. Branch input（当前分支最相关的用户输入片段）
    branch_input = _extract_branch_input(intent, state.get("input_message", ""))
    if not branch_input:
        branch_input = state.get("input_message", "")

    return BranchPromptBundle(
        branch_name=branch_name,
        intent=intent,
        branch_input=branch_input,
        system_context=system_context,
        shared_context=shared_context,
        branch_context=branch_context,
        extra_sections=extra_sections,
        conversation_window=conv_window,
    )


def _split_input_clauses(text: str) -> List[str]:
    """把用户输入切成若干短语，便于按意图抽取分支输入。"""
    import re

    if not text:
        return []

    parts = re.split(r"[。！？!?；;\n]+", text)
    clauses: List[str] = []
    for part in parts:
        for piece in re.split(r"[，,、]", part):
            clause = piece.strip()
            if clause:
                clauses.append(clause)
    return clauses


def _extract_branch_input(intent: str, text: str) -> str:
    """按 intent 抽取最相关的输入片段，降低分支间上下文污染。"""
    if not text:
        return ""

    clauses = _split_input_clauses(text)
    if not clauses:
        return text.strip()

    intent_keywords = {
        "food": ("吃", "喝", "餐", "饭", "菜", "食", "热量", "蛋白", "碳水", "脂肪", "摄入"),
        "workout": ("跑", "走", "骑", "游", "练", "训", "拉伸", "放松", "恢复", "运动", "健身", "消耗", "锻炼"),
        "recipe": ("食谱", "推荐", "怎么吃", "搭配", "菜单", "晚餐", "午餐", "早餐", "加餐"),
        "stats_query": ("统计", "总量", "累计", "今日", "今天", "剩余", "消耗", "摄入"),
    }

    keywords = intent_keywords.get(intent, ())
    matched: List[str] = []

    for clause in clauses:
        if any(keyword in clause for keyword in keywords):
            matched.append(clause)

    # workout 场景里，"如何拉伸/怎么放松" 往往和运动一起出现，保留这类恢复建议
    if intent == "workout":
        for clause in clauses:
            if ("如何" in clause or "怎么" in clause) and any(
                token in clause for token in ("拉伸", "放松", "恢复", "练", "运动")
            ):
                if clause not in matched:
                    matched.append(clause)

    if matched:
        # 保留原始顺序，去重后拼接
        seen = set()
        ordered = []
        for clause in clauses:
            if clause in matched and clause not in seen:
                seen.add(clause)
                ordered.append(clause)
        return "，".join(ordered)

    # 未命中时，适度缩短整句，避免把无关内容完整塞给分支
    return clauses[0] if len(clauses) == 1 else "，".join(clauses[:2])


# ============ 便捷函数 ============

_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager
