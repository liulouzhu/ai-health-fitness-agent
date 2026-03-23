import sys
sys.path.insert(0, ".")

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime

from agent.graph import create_workflow, default_checkpointer
from agent.memory_agent import get_memory_agent
from agent.router_agent import RouterAgent
from agent.llm import get_llm

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

# 全局 app（暂时禁用 checkpointer 以排查流式输出问题）
app_obj = create_workflow(checkpointer=default_checkpointer)
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


# ============ 测试流式接口 ============

@app.get("/test/stream")
async def test_stream():
    """测试流式输出"""
    async def generate():
        text = "这是一段测试文字，用来验证流式输出是否正常工作。每个字之间有延迟。"
        for char in text:
            yield f"data: {char}\n\n"
            import asyncio
            await asyncio.sleep(0.05)  # 50ms 延迟
        yield f"data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")


# ============ 对话接口 ============

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式对话接口 - 直接流式输出 LLM token"""
    config = {"configurable": {"thread_id": "default"}}

    # 获取上下文信息
    restored_state = {}
    try:
        current_state = app_obj.get_state(config=config)
        if current_state is not None:
            state_data = getattr(current_state, 'values', None)
            if state_data and isinstance(state_data, dict):
                restored_state = state_data
    except Exception:
        restored_state = {}

    messages = restored_state.get("messages", [])
    profile_complete = restored_state.get("profile_complete", True)
    last_intent = restored_state.get("last_intent")

    # 先分类意图
    classify_state = {
        "input_message": request.message,
        "messages": messages,
        "profile_complete": profile_complete,
        "last_intent": last_intent,
        "image_info": {"has_image": request.image_url is not None, "image_url": request.image_url}
    }
    classify_state = RouterAgent().classify_intent(classify_state)
    intent = classify_state.get("intent", "general")

    async def generate():
        nonlocal messages, profile_complete, last_intent

        try:
            llm = get_llm()

            # 发送意图
            yield f"data: {{\"intent\": \"{intent}\"}}\n\n"

            # 如果是食物报告或食物查询，走完整的 food_agent 流程
            if intent in ("food_report", "food"):
                food_state = {
                    "input_message": request.message,
                    "messages": list(messages),  # 传副本避免污染原 state
                    "image_info": {"has_image": request.image_url is not None, "image_url": request.image_url},
                    "intent": intent
                }
                from agent.food_agent import FoodAgent
                food_agent = FoodAgent()
                food_state = food_agent.run(food_state)
                response_text = food_state.get("response", "")
                for char in response_text:
                    yield f"data: {char}\n\n"
                yield f"data: [DONE]\n\n"

                # 内存历史持久化
                memory_agent.add_conversation_turn(request.message, response_text)
                memory_agent.extract_and_save_preferences(request.message)
                if memory_agent.should_summarize(threshold=10):
                    memory_agent.summarize_conversations()

                # 更新 LangGraph 状态
                messages = food_state.get("messages", messages)
                last_intent = intent
                _update_graph_state(config, messages, last_intent, profile_complete)
                return

            # 如果是运动报告或运动查询
            if intent in ("workout_report", "workout"):
                workout_state = {
                    "input_message": request.message,
                    "messages": list(messages),  # 传副本
                    "intent": intent
                }
                from agent.workout_agent import WorkoutAgent
                workout_agent = WorkoutAgent()
                workout_state = workout_agent.run(workout_state)
                response_text = workout_state.get("response", "")
                for char in response_text:
                    yield f"data: {char}\n\n"
                yield f"data: [DONE]\n\n"

                memory_agent.add_conversation_turn(request.message, response_text)
                memory_agent.extract_and_save_preferences(request.message)
                if memory_agent.should_summarize(threshold=10):
                    memory_agent.summarize_conversations()

                # 更新 LangGraph 状态
                messages = workout_state.get("messages", messages)
                last_intent = intent
                _update_graph_state(config, messages, last_intent, profile_complete)
                return

            if not profile_complete:
                # 档案不完整，询问用户
                initial_response = memory_agent.get_initial_questions()
                for char in initial_response:
                    yield f"data: {char}\n\n"
                yield f"data: [DONE]\n\n"
                last_intent = intent
                _update_graph_state(config, messages, last_intent, profile_complete)
                return

            # 构建对话历史字符串
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[-10:]])

            # 获取用户偏好
            preferences = memory_agent.get_preferences_for_context()
            if not preferences:
                preferences = "（暂无偏好记录）"

            # 获取长期记忆
            longterm_memory = memory_agent.get_longterm_memory_context(limit=3)
            if not longterm_memory:
                longterm_memory = "（暂无长期记忆）"

            prompt = f"""你是一个健身健康智能助手，可以进行日常对话。

用户偏好（请在回复中注意）：
{preferences}

长期记忆（请在回复中参考）：
{longterm_memory}

对话历史：
{history_str}

用户：{request.message}

请回复用户，保持对话连贯性。如果用户问到健身或饮食相关问题，可以适当引导。注意根据用户偏好选择合适的食物和运动建议。"""

            # 使用 LLM 流式输出
            full_response = ""
            llm_messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": request.message}
            ]

            for chunk in llm.stream(llm_messages):
                if chunk.content:
                    full_response += chunk.content
                    yield f"data: {chunk.content}\n\n"

            yield f"data: [DONE]\n\n"

            # 保存对话历史到 memory
            memory_agent.add_conversation_turn(request.message, full_response)
            memory_agent.extract_and_save_preferences(request.message)
            if memory_agent.should_summarize(threshold=10):
                memory_agent.summarize_conversations()

            # 更新 LangGraph 状态
            messages = messages + [
                {"role": "user", "content": request.message},
                {"role": "assistant", "content": full_response}
            ]
            last_intent = intent
            _update_graph_state(config, messages, last_intent, profile_complete)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"错误: {str(e)}"
            yield f"data: {error_msg}\n\n"
            yield f"data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _update_graph_state(config: dict, messages: list, last_intent: str, profile_complete: bool, app=None):
    """将更新后的状态写回 LangGraph checkpointer（同步调用）

    Args:
        config: LangGraph config dict with thread_id
        messages: updated messages list
        last_intent: last intent string
        profile_complete: whether profile is complete
        app: optional LangGraph app instance (defaults to module-level app_obj)
    """
    try:
        target_app = app if app is not None else app_obj
        target_app.update_state(
            config,
            {
                "messages": messages,
                "last_intent": last_intent,
                "profile_complete": profile_complete,
            }
        )
    except Exception as e:
        print(f"[Warning] Failed to update graph state: {e}")


@app.post("/chat")
async def chat(request: ChatRequest):
    """与智能助手对话"""
    try:
        config = {"configurable": {"thread_id": "default"}}

        # 尝试获取已保存的状态
        restored_state = {}
        try:
            current_state = app_obj.get_state(config=config)
            if current_state is not None:
                state_data = getattr(current_state, 'values', None)
                if state_data and isinstance(state_data, dict):
                    restored_state = state_data
        except Exception:
            restored_state = {}

        # 构建状态
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

        # 使用 invoke（同步）
        result = app_obj.invoke(state, config=config)
        response_text = result.get("response", "")
        intent = result.get("intent")

        # 长期记忆处理
        memory_agent.add_conversation_turn(request.message, response_text)
        memory_agent.extract_and_save_preferences(request.message)
        if memory_agent.should_summarize(threshold=10):
            memory_agent.summarize_conversations()

        # 返回完整响应
        return {"response": response_text, "intent": intent}

    except Exception as e:
        import traceback
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

    # 空更新保护
    if not message:
        return {"message": "没有要更新的字段，请至少传入身高、体重、年龄、性别或目标之一。", "profile": None}

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
