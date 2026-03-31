import sys
sys.path.insert(0, ".")
import uuid
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime
from agent.graph import create_workflow, default_checkpointer
from agent.memory_agent import get_memory_agent
from agent.router_agent import RouterAgent
from agent.llm import get_llm
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

@asynccontextmanager
async def lifespan(app):
    """启动时预热检索器，避免第一次请求时加载 LLM/jieba/BM25 阻塞"""
    import threading

    def _warmup():
        from tools.retriever import get_hybrid_retriever
        print("[Startup] 预热检索器...")
        get_llm()  # 预加载 LLM（QueryRewriter 依赖）
        import jieba
        list(jieba.cut("预热"))  # 预加载 jieba（BM25Okapi 依赖）
        retriever = get_hybrid_retriever("fitness_guide")
        retriever.retrieve("预热")  # 预初始化 HybridRetriever（加载 BM25 索引）
        print("[Startup] 检索器预热完成")

    threading.Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(
    title="健身健康智能助手 API",
    description="基于 LangGraph 的健身健康智能体，提供食物营养分析、食谱推荐、健身指导等功能",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 workflow
app_obj = create_workflow(checkpointer=default_checkpointer)
memory_agent = get_memory_agent()

# 上传目录
UPLOAD_DIR = Path("uploads/images")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 挂载上传目录为静态文件
app.mount("/uploads/images", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# 允许的图片格式
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


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
    intents = classify_state.get("intents", [intent])

    async def generate():
        nonlocal messages, profile_complete, last_intent

        # 初始化流式状态（用于摘要缓冲）
        stream_state = {
            "summary_buffer": restored_state.get("summary_buffer", []),
            "turn_count": restored_state.get("turn_count", 0),
            "last_summary_turn": restored_state.get("last_summary_turn", 0),
        }

        try:
            llm = get_llm()

            # 发送意图
            yield f"data: {{\"intent\": \"{intent}\"}}\n\n"

            # 多意图处理：food + workout / food + stats / workout + stats
            if len(intents) > 1:
                has_food = "food" in intents or "food_report" in intents
                has_workout = "workout" in intents or "workout_report" in intents
                has_stats = "stats_query" in intents

                if has_food and has_workout:
                    # 并发执行 food + workout
                    from agent.food_agent import FoodAgent
                    from agent.workout_agent import WorkoutAgent
                    from concurrent.futures import ThreadPoolExecutor

                    food_state = {
                        "input_message": request.message,
                        "messages": list(messages),
                        "image_info": {"has_image": request.image_url is not None, "image_url": request.image_url},
                        "intent": "food_report" if "food_report" in intents else ("food" if "food" in intents else intent)
                    }
                    workout_state = {
                        "input_message": request.message,
                        "messages": list(messages),
                        "intent": "workout_report" if "workout_report" in intents else ("workout" if "workout" in intents else intent)
                    }

                    with ThreadPoolExecutor(max_workers=2) as executor:
                        food_future = executor.submit(FoodAgent().run, food_state)
                        workout_future = executor.submit(WorkoutAgent().run, workout_state)
                        food_result = food_future.result()
                        workout_result = workout_future.result()

                    responses = []
                    if food_result.get("response"):
                        responses.append(food_result["response"])
                    if workout_result.get("response"):
                        responses.append(workout_result["response"])

                    combined_response = "\n\n".join(responses)
                    for char in combined_response:
                        yield f"data: {char}\n\n"
                    yield f"data: [DONE]\n\n"

                    # 状态-based 摘要缓冲
                    memory_agent.add_conversation_turn(stream_state, request.message, combined_response)
                    memory_agent.extract_and_save_preferences(request.message)
                    if memory_agent.should_summarize(stream_state, threshold=10):
                        memory_agent.summarize_conversations(stream_state)

                    # 合并 messages
                    messages = food_result.get("messages", messages)
                    last_intent = intent
                    _update_graph_state(
                        config, messages, last_intent, profile_complete,
                        summary_buffer=stream_state.get("summary_buffer", []),
                        turn_count=stream_state.get("turn_count", 0),
                        last_summary_turn=stream_state.get("last_summary_turn", 0),
                    )
                    return

                elif has_food and has_stats:
                    from agent.food_agent import FoodAgent
                    from concurrent.futures import ThreadPoolExecutor

                    food_state = {
                        "input_message": request.message,
                        "messages": list(messages),
                        "image_info": {"has_image": request.image_url is not None, "image_url": request.image_url},
                        "intent": "food_report" if "food_report" in intents else intent
                    }

                    with ThreadPoolExecutor(max_workers=2) as executor:
                        food_future = executor.submit(FoodAgent().run, food_state)
                        stats_future = executor.submit(RouterAgent().handle_stats_query, {"input_message": request.message, "messages": list(messages)})
                        food_result = food_future.result()
                        stats_result = stats_future.result()

                    responses = []
                    if food_result.get("response"):
                        responses.append(food_result["response"])
                    if stats_result.get("response"):
                        responses.append(stats_result["response"])

                    combined_response = "\n\n".join(responses)
                    for char in combined_response:
                        yield f"data: {char}\n\n"
                    yield f"data: [DONE]\n\n"

                    # 状态-based 摘要缓冲
                    memory_agent.add_conversation_turn(stream_state, request.message, combined_response)
                    memory_agent.extract_and_save_preferences(request.message)
                    if memory_agent.should_summarize(stream_state, threshold=10):
                        memory_agent.summarize_conversations(stream_state)

                    messages = food_result.get("messages", messages)
                    last_intent = intent
                    _update_graph_state(
                        config, messages, last_intent, profile_complete,
                        summary_buffer=stream_state.get("summary_buffer", []),
                        turn_count=stream_state.get("turn_count", 0),
                        last_summary_turn=stream_state.get("last_summary_turn", 0),
                    )
                    return

                elif has_workout and has_stats:
                    from agent.workout_agent import WorkoutAgent
                    from concurrent.futures import ThreadPoolExecutor

                    workout_state = {
                        "input_message": request.message,
                        "messages": list(messages),
                        "intent": "workout_report" if "workout_report" in intents else intent
                    }

                    with ThreadPoolExecutor(max_workers=2) as executor:
                        workout_future = executor.submit(WorkoutAgent().run, workout_state)
                        stats_future = executor.submit(RouterAgent().handle_stats_query, {"input_message": request.message, "messages": list(messages)})
                        workout_result = workout_future.result()
                        stats_result = stats_future.result()

                    responses = []
                    if workout_result.get("response"):
                        responses.append(workout_result["response"])
                    if stats_result.get("response"):
                        responses.append(stats_result["response"])

                    combined_response = "\n\n".join(responses)
                    for char in combined_response:
                        yield f"data: {char}\n\n"
                    yield f"data: [DONE]\n\n"

                    # 状态-based 摘要缓冲
                    memory_agent.add_conversation_turn(stream_state, request.message, combined_response)
                    memory_agent.extract_and_save_preferences(request.message)
                    if memory_agent.should_summarize(stream_state, threshold=10):
                        memory_agent.summarize_conversations(stream_state)

                    messages = workout_result.get("messages", messages)
                    last_intent = intent
                    _update_graph_state(
                        config, messages, last_intent, profile_complete,
                        summary_buffer=stream_state.get("summary_buffer", []),
                        turn_count=stream_state.get("turn_count", 0),
                        last_summary_turn=stream_state.get("last_summary_turn", 0),
                    )
                    return

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

                # 状态-based 摘要缓冲
                memory_agent.add_conversation_turn(stream_state, request.message, response_text)
                memory_agent.extract_and_save_preferences(request.message)
                if memory_agent.should_summarize(stream_state, threshold=10):
                    memory_agent.summarize_conversations(stream_state)

                # 更新 LangGraph 状态
                messages = food_state.get("messages", messages)
                last_intent = intent
                _update_graph_state(
                    config, messages, last_intent, profile_complete,
                    summary_buffer=stream_state.get("summary_buffer", []),
                    turn_count=stream_state.get("turn_count", 0),
                    last_summary_turn=stream_state.get("last_summary_turn", 0),
                )
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

                # 状态-based 摘要缓冲
                memory_agent.add_conversation_turn(stream_state, request.message, response_text)
                memory_agent.extract_and_save_preferences(request.message)
                if memory_agent.should_summarize(stream_state, threshold=10):
                    memory_agent.summarize_conversations(stream_state)

                # 更新 LangGraph 状态
                messages = workout_state.get("messages", messages)
                last_intent = intent
                _update_graph_state(
                    config, messages, last_intent, profile_complete,
                    summary_buffer=stream_state.get("summary_buffer", []),
                    turn_count=stream_state.get("turn_count", 0),
                    last_summary_turn=stream_state.get("last_summary_turn", 0),
                )
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

            # 状态-based 摘要缓冲
            memory_agent.add_conversation_turn(stream_state, request.message, full_response)
            memory_agent.extract_and_save_preferences(request.message)
            if memory_agent.should_summarize(stream_state, threshold=10):
                memory_agent.summarize_conversations(stream_state)

            # 更新 LangGraph 状态
            messages = messages + [
                {"role": "user", "content": request.message},
                {"role": "assistant", "content": full_response}
            ]
            last_intent = intent
            _update_graph_state(
                config, messages, last_intent, profile_complete,
                summary_buffer=stream_state.get("summary_buffer", []),
                turn_count=stream_state.get("turn_count", 0),
                last_summary_turn=stream_state.get("last_summary_turn", 0),
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"错误: {str(e)}"
            yield f"data: {error_msg}\n\n"
            yield f"data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _update_graph_state(
    config: dict,
    messages: list,
    last_intent: str,
    profile_complete: bool,
    summary_buffer: list = None,
    turn_count: int = None,
    last_summary_turn: int = None,
    app=None
):
    """将更新后的状态写回 LangGraph checkpointer（同步调用）

    Args:
        config: LangGraph config dict with thread_id
        messages: updated messages list
        last_intent: last intent string
        profile_complete: whether profile is complete
        summary_buffer: optional summary buffer list
        turn_count: optional turn count
        last_summary_turn: optional last summary turn
        app: optional LangGraph app instance (defaults to module-level app_obj)
    """
    try:
        target_app = app if app is not None else app_obj
        updates = {
            "messages": messages,
            "last_intent": last_intent,
            "profile_complete": profile_complete,
        }
        if summary_buffer is not None:
            updates["summary_buffer"] = summary_buffer
        if turn_count is not None:
            updates["turn_count"] = turn_count
        if last_summary_turn is not None:
            updates["last_summary_turn"] = last_summary_turn
        target_app.update_state(config, updates)
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

        # 构建状态（包含新的摘要缓冲字段）
        state = {
            "input_message": request.message,
            "image_info": {
                "has_image": request.image_url is not None,
                "image_url": request.image_url
            },
            "messages": restored_state.get("messages", []),
            "last_intent": restored_state.get("last_intent"),
            "profile_complete": restored_state.get("profile_complete", True),
            "summary_buffer": restored_state.get("summary_buffer", []),
            "turn_count": restored_state.get("turn_count", 0),
            "last_summary_turn": restored_state.get("last_summary_turn", 0),
        }

        # 使用 invoke（同步）
        result = app_obj.invoke(state, config=config)
        response_text = result.get("response", "")
        intent = result.get("intent")

        # 长期记忆处理（基于 state 的摘要缓冲）
        memory_agent.add_conversation_turn(result, request.message, response_text)
        memory_agent.extract_and_save_preferences(request.message)
        if memory_agent.should_summarize(result, threshold=10):
            memory_agent.summarize_conversations(result)

        # 将更新后的摘要缓冲写回 checkpointer
        _update_graph_state(
            config,
            result.get("messages", []),
            result.get("last_intent"),
            result.get("profile_complete", True),
            summary_buffer=result.get("summary_buffer", []),
            turn_count=result.get("turn_count", 0),
            last_summary_turn=result.get("last_summary_turn", 0),
        )

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
    """获取对话历史（从 LangGraph state 的 messages 窗口读取，聚合为 user/agent 配对）"""
    config = {"configurable": {"thread_id": "default"}}
    try:
        current_state = app_obj.get_state(config=config)
        if current_state is not None:
            state_data = getattr(current_state, 'values', None)
            if state_data and isinstance(state_data, dict):
                messages = state_data.get("messages", [])
                # 从最老开始，每相邻 user+assistant 配成一轮
                turns = []
                i = 0
                while i + 1 < len(messages):
                    user_msg = messages[i]
                    assistant_msg = messages[i + 1]
                    if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
                        turns.append({
                            "timestamp": "",
                            "user": user_msg.get("content", ""),
                            "agent": assistant_msg.get("content", ""),
                        })
                        i += 2
                    else:
                        i += 1

                # 返回最近 20 轮（按时间正序，与前端渲染顺序一致）
                return {"history": turns[-20:]}
    except Exception:
        pass
    return {"history": []}


@app.delete("/history")
async def clear_conversation_history():
    """清除对话历史（清空 LangGraph state 中的短期会话字段，保留 profile_complete）"""
    config = {"configurable": {"thread_id": "default"}}
    try:
        # 步骤1：优先从 checkpointer state 读取
        current_profile_complete = None
        try:
            state_obj = app_obj.get_state(config=config)
            if state_obj is not None:
                values = getattr(state_obj, 'values', None)
                if values and isinstance(values, dict):
                    current_profile_complete = values.get("profile_complete")
        except Exception:
            pass

        # 步骤2：checkpointer 无 state，从用户档案重新计算
        if current_profile_complete is None:
            current_profile_complete = memory_agent.is_profile_complete()

        _update_graph_state(
            config,
            messages=[],
            last_intent=None,
            profile_complete=current_profile_complete,
            summary_buffer=[],
            turn_count=0,
            last_summary_turn=0,
        )
    except Exception as e:
        print(f"[Warning] Failed to clear history: {e}")
    return {"message": "对话历史已清除"}


# ============ 图片上传接口 ============

class UploadResponse(BaseModel):
    success: bool
    filename: str
    image_url: str
    data_url: str
    content_type: str
    size: int


@app.post("/upload-image", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """上传本地图片，返回可用于聊天接口的 image_url"""
    # 1. 校验扩展名
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的图片格式。支持的格式: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # 2. 校验 MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 MIME 类型。支持的类型: {', '.join(ALLOWED_MIME_TYPES)}"
        )

    # 3. 读取并校验文件大小
    contents = await file.read()
    size = len(contents)
    if size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超过限制 ({MAX_FILE_SIZE // (1024*1024)}MB)"
        )
    if size == 0:
        raise HTTPException(status_code=400, detail="文件为空")

    # 4. 生成唯一文件名
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = UPLOAD_DIR / unique_name

    # 5. 保存文件
    with open(file_path, "wb") as f:
        f.write(contents)

    # 6. 构建 image_url（相对路径，供聊天接口使用）
    image_url = f"/uploads/images/{unique_name}"

    # 7. 构建 data_url（base64，供多模态模型直接使用）
    data_url = f"data:{file.content_type};base64,{base64.b64encode(contents).decode('utf-8')}"

    return UploadResponse(
        success=True,
        filename=unique_name,
        image_url=image_url,
        data_url=data_url,
        content_type=file.content_type,
        size=size
    )


# ============ 健康检查 ============

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
