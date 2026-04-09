from typing import Annotated, Dict, List, Optional, TypedDict
import operator


# ============ 用户长期档案 ============
class UserProfile(TypedDict):
    """用户基础信息和长期目标"""
    user_id: str
    height: float           # cm
    weight: float           # kg
    age: int
    gender: str             # male/female
    goal: str               # 减脂/增肌/维持
    target_calories: int    # 每日目标热量
    target_protein: int     # 每日目标蛋白质(g)


# ============ 每日统计 ============
class DailyStats(TypedDict):
    """每日摄入和消耗统计"""
    date: str               # YYYY-MM-DD
    consumed_calories: int  # 已摄入热量
    consumed_protein: float # 已摄入蛋白质(g)
    consumed_fat: float     # 已摄入脂肪(g)
    consumed_carbs: float   # 已摄入碳水(g)
    burned_calories: int    # 运动消耗热量
    meals: List[Dict]       # 餐食记录 [{name, calories, protein, time}]
    workouts: List[Dict]    # 运动记录 [{type, duration, calories, time}]


# ============ 消息和图片 ============
class ImageInfo(TypedDict):
    """用户上传的图片信息"""
    image_url: Optional[str]    # 图片URL或base64
    has_image: bool             # 是否包含图片


# ============ 待确认的候选记录（确认流程唯一主状态）============
class PendingConfirmation(TypedDict, total=False):
    """确认流程唯一主状态。

    所有待确认的候选数据（食物/运动记录）都放在这里，
    不在 AgentState 顶层重复存放 candidate_meal / candidate_workout。
    """
    action: str                      # "log_meal" / "log_workout" / "log_both"
    candidate_meal: Optional[Dict]    # 食物候选记录（仅 action=log_meal/log_both 时有效）
    candidate_workout: Optional[Dict] # 运动候选记录（仅 action=log_workout/log_both 时有效）
    analysis_text: str               # 分析文本（展示给用户）
    confirmed: Optional[bool]        # None=pending, True=confirmed, False=cancelled


# ============ LangGraph 主状态 ============
#
# 字段分组：
# 1. 输入：input_message / image_info
# 2. 路由：intent / intents / last_intent / route_decision
# 3. 记忆镜像：user_profile / daily_stats / profile_complete
#    （从 memory_agent 同步而来，非主数据源；主数据源在 memory 模块）
# 4. 对话历史：messages / summary_buffer / turn_count / last_summary_turn
# 5. 确认流程：pending_confirmation / requires_confirmation
#    （pending_confirmation 是唯一主状态；requires_confirmation 由其派生，仅作路由快捷标记）
# 6. fan-out 分支结果：food_branch_result / workout_branch_result / stats_branch_result
#                       food_pending_conf / workout_pending_conf
# 7. 最终输出：final_response / response / food_result / workout_result / stats_result / recipe_result
# 8. legacy 兼容层：pending_stats / pending_response
#    （仅旧版 fallback，不主导主流程；后续逐步清理）
class AgentState(TypedDict):
    # --- 输入 ---
    input_message: str          # 用户文字输入
    image_info: ImageInfo        # 图片信息

    # --- 路由决策（由 classify_intent 节点填充） ---
    intent: str                  # food / workout / recipe / stats_query / profile_update / confirm / general
    intents: List[str]           # 多意图列表，如 ["food", "workout"]
    last_intent: Optional[str]   # 上一个意图，用于上下文推断
    route_decision: Optional[str]  # 路由到的目标节点名（调试/可观测性）

    # --- 记忆镜像（memory 模块是主数据源，此处仅为本轮运行时方便缓存）---
    user_profile: Optional[UserProfile]     # 用户档案镜像（主数据源在 memory_agent）
    daily_stats: Optional[DailyStats]       # 今日统计镜像（主数据源在 memory_agent）
    profile_complete: bool                  # 用户档案是否完整（路由缓存，非持久真相）

    # --- 对话历史（Annotated list，用 operator.add 累加）---
    messages: Annotated[list, operator.add]

    # --- 摘要缓冲（用于长期记忆写入） ---
    summary_buffer: List[Dict]        # 未摘要的对话轮次 [{"user": ..., "agent": ..., "timestamp": ...}]
    turn_count: int                   # 当前会话总轮次
    last_summary_turn: int            # 上次写入长期记忆时的 turn_count

    # --- 确认流程 ---
    # pending_confirmation 是唯一主状态；requires_confirmation 是派生字段，仅作路由快捷标记。
    # 路由判断统一用 pending_confirmation.confirmed 是否为 None，不依赖 requires_confirmation。
    pending_confirmation: PendingConfirmation  # 确认流程唯一主状态
    requires_confirmation: bool                # 派生字段：是否有待确认内容（快捷标记）

    # --- fan-out 分支结果（每个分支写自己专属字段，避免并发覆盖） ---
    food_branch_result: Optional[str]     # food_branch 写入的原始输出
    workout_branch_result: Optional[str]  # workout_branch 写入的原始输出
    stats_branch_result: Optional[str]    # stats_branch 写入的原始输出
    food_pending_conf: Optional[PendingConfirmation]  # food_branch 写入的待确认上下文
    workout_pending_conf: Optional[PendingConfirmation]  # workout_branch 写入的待确认上下文

    # --- 最终响应 ---
    final_response: Optional[str]     # 最终回复给用户的内容（join 节点合并后写入）
    response: Optional[str]           # 各叶子节点写入的响应（join 节点合并后清空）

    # --- 子Agent原始结果（单意图节点写入，join 节点读取后清理） ---
    food_result: Optional[str]       # food agent 原始输出
    workout_result: Optional[str]    # workout agent 原始输出
    stats_result: Optional[str]      # stats agent 原始输出
    recipe_result: Optional[str]     # recipe agent 原始输出

    # --- legacy 兼容层（仅旧版 fallback，不主导主流程） ---
    pending_stats: Optional[Dict]        # 旧版 pending_stats（兼容旧 pending_stats.json fallback）
    pending_response: Optional[str]       # 旧版 pending_response（已废弃，仅保留字段防止 KeyError）
