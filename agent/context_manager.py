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
from agent.memory_agent import get_memory_agent
from config import AgentConfig

logger = logging.getLogger(__name__)

# ============ 分层数据结构 ============

class ContextBundle(TypedDict, total=False):
    """统一上下文数据结构，所有字段均按语义分层"""
    system_context: str              # system prompt 文本
    extra_context: str               # intent-specific 额外上下文片段
    conversation_window: List[Dict]  # 运行时滑动窗口消息
    user_memory: Dict                # profile + preferences (raw dict)
    task_context: Dict              # intent-specific 业务上下文
    retrieved_knowledge: str        # 外部检索内容


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
            logger.debug(f"[TokenEstimator] tiktoken unavailable ({e}), using char-based fallback")
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
            parts.append(f"食物记录：{food.get('name', '?')}（{food.get('calories', 0)} kcal，{food.get('protein', 0)}g 蛋白）")
        if pending.get("workout"):
            workout = pending["workout"]
            parts.append(f"运动记录：{workout.get('type', '?')}（{workout.get('duration', 0)}分钟，{workout.get('calories', 0)} kcal）")
    elif ptype == "meal":
        data = pending.get("data", {})
        parts.append(f"食物记录：{data.get('name', '?')}（{data.get('calories', 0)} kcal，{data.get('protein', 0)}g 蛋白）")
    elif ptype == "workout":
        data = pending.get("workout", pending.get("data", {}))
        parts.append(f"运动记录：{data.get('type', '?')}（{data.get('duration', 0)}分钟，{data.get('calories', 0)} kcal）")

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
直接回复健身指导内容，不需要额外解释。""",

    "recipe": """你是一个营养师。根据用户的饮食目标和限制，推荐合适的食谱。

请根据以上信息，推荐合适的食谱组合，确保：
1. 总热量不超过剩余热量
2. 蛋白质尽量达到目标
3. 食物种类多样化
4. **严格避免推荐用户不喜欢或过敏的食物**

直接回复推荐内容，不需要额外解释。""",

    "general": """你是一个健身健康智能助手，可以进行日常对话。
请回复用户，保持对话连贯性。如果用户问到健身或饮食相关问题，可以适当引导。""",

    "classify_intent": """你是一个智能路由器，负责判断用户意图。

可选意图：food, workout, recipe, stats_query, profile_update, confirm, general, food_report, workout_report

重要：一个用户输入可能包含多个意图，以逗号分隔返回。
只返回意图标签，不要任何解释。""",
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
                extra_text = _truncate_to_tokens(extra_text, cfg.MAX_EXTRA_CONTEXT_TOKENS, est)
                logger.debug(f"[CtxMgr] extra_context trimmed: {extra_tokens} → {est.estimate(extra_text)} tokens")
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
                    logger.debug(f"[CtxMgr] extra_section '{key}' trimmed: {value_tokens} → {est.estimate(value)} tokens")
                sections[key] = value

        # ----- Section 4: conversation_window（token 预算裁剪） -----
        conv = bundle["conversation_window"]
        # 估算已有 sections 消耗
        consumed = sum(est.estimate(v) for v in sections.values()) + 4  # +4 = user msg overhead
        remaining = cfg.MAX_CONVERSATION_WINDOW_TOKENS
        trimmed_conv = []
        for msg in reversed(conv):
            t = est.estimate(msg.get("content", ""))
            if remaining >= t + 4:
                trimmed_conv.insert(0, msg)
                remaining -= (t + 4)
            else:
                break
        sections["conversation_window"] = trimmed_conv

        # ----- 汇总 token 报告 -----
        total = est.estimate_messages([
            {"role": "system", "content": v} if k != "conversation_window" else v
            for k, v in sections.items()
            if k != "conversation_window"
        ] + list(sections.get("conversation_window", [])))
        logger.debug(
            f"[CtxMgr] intent={intent} | "
            f"system={system_tokens} | extra={est.estimate(sections.get('extra_context',''))} | "
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
                if key not in ("system_context", "extra_context", "conversation_window") and value:
                    extra_all.append(f"{key}：{value}")

        if extra_all:
            messages.append({"role": "system", "content": "\n".join(extra_all)})

        for msg in sections.get("conversation_window", []):
            messages.append(msg)

        messages.append({"role": "user", "content": user_input})
        return messages

    def append_messages(self, state: AgentState, user_msg: str, assistant_msg: str) -> None:
        """
        向 state['messages'] 追加一轮对话，并维持滑动窗口上限。
        """
        messages = state.get("messages", [])
        messages = messages + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        if len(messages) > AgentConfig.MAX_RECENT_MESSAGES:
            messages = messages[-AgentConfig.MAX_RECENT_MESSAGES:]
        state["messages"] = messages

    # ---- 公开工具方法 ----

    def get_preferences_str(self) -> str:
        """
        公开 API：将 preferences 字典格式化为上下文字符串。
        """
        return self._get_preferences_str()

    def format_task_context(self, intent: str, task_context: Optional[Dict] = None) -> str:
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
        return messages[-AgentConfig.MAX_RECENT_MESSAGES:]

    def _get_user_memory(self) -> Dict:
        """加载用户长期记忆：profile + preferences"""
        profile = self.memory.load_profile()
        preferences = self.memory.load_preferences()
        return {"profile": profile, "preferences": preferences}

    def _build_task_context(self, intent: str, state: AgentState) -> Dict:
        """
        按 intent 加载不同的业务上下文
        """
        ctx: Dict[str, Any] = {}

        if intent in ("food", "food_report"):
            today = self.memory.load_daily_stats()
            profile = self.memory.load_profile()
            ctx["daily_stats"] = today
            ctx["profile"] = profile

        elif intent in ("workout", "workout_report"):
            prefs = self.memory.load_preferences()
            today = self.memory.load_daily_stats()
            ctx["workout_preferences"] = prefs.get("workout_preferences", {})
            ctx["daily_stats"] = today

        elif intent == "recipe":
            profile = self.memory.load_profile()
            today = self.memory.load_daily_stats()
            prefs = self.memory.load_preferences()
            target_cal = int(profile.get("target_calories", 2000))
            target_pro = int(profile.get("target_protein", 100))
            consumed_cal = today.get("consumed_calories", 0)
            consumed_pro = today.get("consumed_protein", 0)
            burned_cal = today.get("burned_calories", 0)
            ctx.update({
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
            })

        elif intent == "stats_query":
            today = self.memory.load_daily_stats()
            profile = self.memory.load_profile()
            ctx["daily_stats"] = today
            ctx["profile"] = profile

        elif intent == "confirm":
            pending = self.memory.load_pending_stats()
            ctx["pending_stats"] = pending

        return ctx

    def _build_extra_context(self, intent: str, task_context: Dict) -> str:
        """
        在 system prompt 之后、history 之前，注入意图特定的额外上下文片段。
        返回空字符串表示不需要额外注入。
        """
        parts = []
        prefs_str = self._get_preferences_str()
        longterm = self.memory.get_longterm_memory_context(
            limit=AgentConfig.MAX_LONGTERM_MEMORY_ITEMS
        )

        if intent == "general":
            if prefs_str:
                parts.append(f"用户偏好：{prefs_str}")
            if longterm:
                parts.append(f"长期记忆：{longterm}")

        elif intent in ("food", "food_report"):
            stats = task_context.get("daily_stats", {})
            if stats:
                parts.append(self._fmt_daily_intake(stats))

        elif intent in ("workout", "workout_report"):
            parts.extend(self._fmt_workout_prefs(task_context.get("workout_preferences", {})))

        elif intent == "confirm":
            pending = task_context.get("pending_stats")
            if pending:
                parts.append(_format_pending_stats(pending))

        return "\n".join(parts)

    def _format_task_context_text(self, intent: str, task_context: Dict) -> str:
        """将 task_context 格式化为易读文本（用于 agent 的 extra_sections）"""
        parts = []

        if intent in ("food", "food_report"):
            stats = task_context.get("daily_stats", {})
            if stats:
                parts.append(self._fmt_daily_intake(stats))

        elif intent in ("workout", "workout_report"):
            parts.extend(self._fmt_workout_prefs(task_context.get("workout_preferences", {})))
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
                target_cal = int(profile.get("target_calories", 2000))
                target_pro = int(profile.get("target_protein", 100))
                remaining = max(0, target_cal - stats.get("consumed_calories", 0) + stats.get("burned_calories", 0))
                parts.append(f"今日摄入：{int(stats.get('consumed_calories', 0))} / {target_cal} kcal")
                parts.append(f"今日蛋白：{stats.get('consumed_protein', 0):.0f} / {target_pro} g")
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


# ============ 便捷函数 ============

_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager
