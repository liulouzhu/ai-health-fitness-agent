from typing import Annotated, Any, Dict, List, Optional, TypedDict
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


# ============ 待确认的候选记录 ============
class PendingConfirmation(TypedDict, total=False):
    """待确认的候选记录（用于 graph-native 确认流程）"""
    action: str                    # "log_meal" / "log_workout" / "log_both"
    candidate_meal: Optional[Dict]  # 食物候选记录
    candidate_workout: Optional[Dict]  # 运动候选记录
    analysis_text: str             # 分析文本（展示给用户）
    confirmed: Optional[bool]      # None=pending, True=confirmed, False=cancelled


# ============ LangGraph 主状态 ============
class AgentState(TypedDict):
    # --- 输入 ---
    input_message: str          # 用户文字输入
    image_info: ImageInfo        # 图片信息

    # --- 路由决策（由 classify_intent 节点填充） ---
    intent: str                  # food / workout / recipe / stats_query / profile_update / confirm / general
    intents: List[str]           # 多意图列表，如 ["food", "workout"]
    last_intent: Optional[str]   # 上一个意图，用于上下文推断
    route_decision: Optional[str]  # 路由到的目标节点名（调试/可观测性）

    # --- 用户数据 ---
    user_profile: UserProfile     # 用户档案(长期记忆)
    daily_stats: DailyStats      # 今日统计(每日记忆)
    profile_complete: bool       # 用户档案是否完整

    # --- 对话历史（Annotated list，用 operator.add 累加）---
    messages: Annotated[list, operator.add]

    # --- 摘要缓冲（用于长期记忆写入） ---
    summary_buffer: List[Dict]        # 未摘要的对话轮次 [{"user": ..., "agent": ..., "timestamp": ...}]
    turn_count: int                   # 当前会话总轮次
    last_summary_turn: int            # 上次写入长期记忆时的 turn_count

    # --- 候选记录（确认流程用） ---
    candidate_meal: Optional[Dict]    # {"name": ..., "calories": ..., "protein": ..., ...}
    candidate_workout: Optional[Dict] # {"type": ..., "duration": ..., "calories": ...}
    pending_confirmation: PendingConfirmation  # 当前待确认状态

    # --- 确认流程控制 ---
    requires_confirmation: bool       # 是否需要用户确认才能 commit
    pending_action: Optional[str]     # "log_meal" / "log_workout" / "log_both" / None

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

    # --- 待确认数据（兼容旧逻辑，逐步迁移） ---
    pending_stats: Optional[Dict]        # 旧版 pending_stats（迁移期间保留）
    pending_response: Optional[str]       # 旧版 pending_response
