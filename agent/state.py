from typing import Annotated, Any, Dict, List, Optional, TypedDict
import operator


def merge_results(prev: Any, new: Any) -> Any:
    """合并两个结果，支持多意图场景

    Args:
        prev: 之前的结果
        new: 新的结果
    Returns:
        合并后的结果
    """
    if new is None:
        return prev
    if prev is None:
        return new
    # 多个结果用分隔符连接
    return f"{prev}\n\n---\n\n{new}"



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
    messages: Annotated[list, operator.add]  # 对话历史

    # --- 输出（支持并发归并）---
    response: Annotated[Optional[str], merge_results]  # 最终回复给用户的内容

    # --- 子Agent结果（支持并发归并）---
    food_result: Annotated[Optional[str], merge_results]   # food agent 原始输出
    workout_result: Annotated[Optional[str], merge_results] # workout agent 原始输出

    # --- 待确认数据（支持并发归并）---
    pending_stats: Annotated[Optional[Dict], merge_results]   # 待确认的统计数据 {"type": "meal"/"workout", "data": {...}}
    pending_response: Annotated[Optional[str], merge_results] # 待确认的原始分析结果
