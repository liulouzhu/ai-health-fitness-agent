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


# ============ LangGraph 主状态 ============
class AgentState(TypedDict):
    # --- 输入 ---
    input_message: str          # 用户文字输入
    image_info: ImageInfo       # 图片信息

    # --- 路由决策 ---
    intent: str                 # food / workout / profile_update / general (单意图，回退用)
    intents: List[str]          # 多意图列表，如 ["food", "workout"]
    last_intent: Optional[str]  # 上一个意图，用于上下文推断
    profile_complete: bool      # 用户档案是否完整

    # --- 用户数据 ---
    user_profile: UserProfile   # 用户档案(长期记忆)
    daily_stats: DailyStats     # 今日统计(每日记忆)

    # --- 对话历史 ---
    messages: Annotated[list, operator.add]  # 对话历史（运行时滑动窗口）

    # --- 摘要缓冲（用于长期记忆写入） ---
    summary_buffer: List[Dict]        # 未摘要的对话轮次 [{"user": ..., "agent": ..., "timestamp": ...}]
    turn_count: int                   # 当前会话总轮次
    last_summary_turn: int            # 上次写入长期记忆时的 turn_count

    # --- 最终响应 ---
    response: Optional[str]     # 最终回复给用户的内容

    # --- 子Agent结果 ---
    food_result: Optional[str]           # food agent 输出
    workout_result: Optional[str]        # workout agent 输出
    recipe_result: Optional[str]          # recipe agent 输出

    # --- 待确认数据 ---
    pending_stats: Optional[Dict]        # 待确认的统计数据
    pending_response: Optional[str]      # 待确认的原始分析结果
