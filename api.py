import sys
sys.path.insert(0, ".")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime

from langgraph.checkpoint.memory import InMemorySaver
from agent.graph import create_workflow
from agent.memory_agent import get_memory_agent

# 创建 FastAPI 应用
app = FastAPI(
    title="健身健康智能助手 API",
    description="基于 LangGraph 的健身健康智能体，提供食物营养分析、食谱推荐、健身指导等功能"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 checkpointer 和 app
checkpointer = InMemorySaver()
app_obj = create_workflow(checkpointer=checkpointer)
memory_agent = get_memory_agent()


# ============ 请求/响应模型 ============

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    image_url: Optional[str] = None

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("消息不能为空")
        return v.strip()


class ChatResponse(BaseModel):
    response: str
    intent: Optional[str] = None


class ProfileRequest(BaseModel):
    height: Optional[float] = Field(None, gt=0, le=300, description="身高(cm)，范围 0-300")
    weight: Optional[float] = Field(None, gt=0, le=500, description="体重(kg)，范围 0-500")
    age: Optional[int] = Field(None, gt=0, le=150, description="年龄，范围 0-150")
    gender: Optional[str] = Field(None, description="性别")
    goal: Optional[str] = Field(None, description="健身目标")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("male", "female", "男", "女"):
            raise ValueError("性别必须是 male/female/男/女")
        return v

    @field_validator("goal")
    @classmethod
    def validate_goal(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("减脂", "增肌", "维持", "cut", "bulk", "maintain"):
            raise ValueError("目标必须是 减脂/增肌/维持 或 cut/bulk/maintain")
        return v


class ProfileResponse(BaseModel):
    user_id: str
    height: float
    weight: float
    age: int
    gender: str
    goal: str
    target_calories: int
    target_protein: int


class DailyStatsResponse(BaseModel):
    date: str
    consumed_calories: int
    consumed_protein: float
    burned_calories: int
    remaining_calories: int
    remaining_protein: float


# ============ 对话接口 ============

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """与智能助手对话"""
    try:
        print(f"\n{'='*60}")
        print(f"[API] 收到新请求: {request.message[:30]}...")
        print(f"{'='*60}")

        config = {"configurable": {"thread_id": "default"}}

        # 尝试获取已保存的状态
        restored_state = {}
        try:
            current_state = app_obj.get_state(config=config)
            if current_state is not None:
                state_data = getattr(current_state, 'values', None)
                if state_data and isinstance(state_data, dict):
                    restored_state = state_data
                    print(f"[DEBUG] 恢复状态: last_intent={state_data.get('last_intent')}, messages数量={len(state_data.get('messages', []))}")
        except Exception as e:
            import traceback
            print(f"[WARN] 获取历史状态失败: {e}")
            traceback.print_exc()

        # 构建状态 - 使用恢复的状态，确保 last_intent 和 messages 不丢失
        state = {
            "input_message": request.message,
            "image_info": {
                "has_image": request.image_url is not None,
                "image_url": request.image_url
            },
            "messages": restored_state.get("messages", []),
            "last_intent": restored_state.get("last_intent"),
            "profile_complete": restored_state.get("profile_complete", True),
        }

        print(f"[DEBUG] 调用 invoke - input: '{request.message[:30]}...', last_intent: {state['last_intent']}, messages数量: {len(state['messages'])}")

        result = app_obj.invoke(state, config=config)

        result_messages = result.get("messages", [])
        print(f"[DEBUG] invoke 完成 - intent: {result.get('intent')}, last_intent: {result.get('last_intent')}, result messages数量: {len(result_messages)}")

        # ============ 长期记忆处理 ============
        response_text = result.get("response", "")

        # 1. 记录对话历史
        memory_agent.add_conversation_turn(request.message, response_text)

        # 2. 从用户消息中提取偏好并保存
        memory_agent.extract_and_save_preferences(request.message)

        # 3. 检查是否需要摘要（每10条消息触发一次）
        if memory_agent.should_summarize(threshold=10):
            summary_result = memory_agent.summarize_conversations()
            if summary_result:
                print(f"[Memory] 对话已摘要: {summary_result.get('summary', '')[:50]}...")

        return ChatResponse(
            response=response_text,
            intent=result.get("intent")
        )
    except Exception as e:
        import traceback
        print("[ERROR] Chat endpoint error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============ 用户档案接口 ============

@app.get("/profile", response_model=ProfileResponse)
async def get_profile():
    """获取用户档案"""
    profile = memory_agent.load_profile()
    if not profile.get("height"):
        raise HTTPException(status_code=404, detail="用户档案不存在")

    return ProfileResponse(
        user_id=profile.get("user_id", "default"),
        height=float(profile.get("height", 0)),
        weight=float(profile.get("weight", 0)),
        age=int(profile.get("age", 0)),
        gender=profile.get("gender", "unknown"),
        goal=profile.get("goal", "unknown"),
        target_calories=int(profile.get("target_calories", 0)),
        target_protein=int(profile.get("target_protein", 0))
    )


@app.post("/profile")
async def create_or_update_profile(request: ProfileRequest):
    """创建或更新用户档案"""
    # 构建更新消息
    updates = []
    if request.height:
        updates.append(f"身高{request.height}")
    if request.weight:
        updates.append(f"体重{request.weight}")
    if request.age:
        updates.append(f"年龄{request.age}")
    if request.gender:
        updates.append(f"性别{request.gender}")
    if request.goal:
        updates.append(f"目标{request.goal}")

    message = "，".join(updates)

    # 判断是创建还是更新
    profile = memory_agent.load_profile()
    if not profile.get("height"):
        # 创建
        result = memory_agent.create_profile(message)
        return {"message": "档案创建成功", "profile": result}
    else:
        # 更新
        result = memory_agent.update_profile(message)
        return {"message": "档案更新成功" if result.get("changed") else "未检测到变化", "profile": result}


# ============ 每日统计接口 ============

@app.get("/daily_stats", response_model=DailyStatsResponse)
async def get_daily_stats():
    """获取今日统计"""
    stats = memory_agent.load_daily_stats()
    profile = memory_agent.load_profile()

    target_cal = int(profile.get("target_calories", 2000))
    target_pro = int(profile.get("target_protein", 100))
    remaining_cal = target_cal - stats.get("consumed_calories", 0) + stats.get("burned_calories", 0)
    remaining_pro = target_pro - stats.get("consumed_protein", 0)

    return DailyStatsResponse(
        date=stats.get("date", datetime.now().strftime("%Y-%m-%d")),
        consumed_calories=int(stats.get("consumed_calories", 0)),
        consumed_protein=float(stats.get("consumed_protein", 0)),
        burned_calories=int(stats.get("burned_calories", 0)),
        remaining_calories=remaining_cal,
        remaining_protein=remaining_pro
    )


@app.get("/history")
async def get_conversation_history():
    """获取对话历史"""
    history = memory_agent.get_conversation_history(limit=20)
    return {"history": history}


@app.delete("/history")
async def clear_conversation_history():
    """清除对话历史"""
    memory_agent.clear_old_conversation_history(keep_recent=0)
    return {"message": "对话历史已清除"}


# ============ 健康检查 ============

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
