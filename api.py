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
from agent.memory import get_memory_agent
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
        get_llm()  # 预加载 LLM
        import jieba
        list(jieba.cut("预热"))  # 预加载 jieba
        retriever = get_hybrid_retriever("fitness_guide")
        retriever.retrieve("预热")
        print("[Startup] 检索器预热完成")

    threading.Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(
    title="健身健康智能助手 API",
    description="基于 LangGraph 的健身健康智能体，提供食物营养分析、食谱推荐、健身指导等功能",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 workflow（使用 PostgreSQL checkpointer）
app_obj = create_workflow(checkpointer=default_checkpointer)
memory_agent = get_memory_agent()

# 上传目录
UPLOAD_DIR = Path("uploads/images")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads/images", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


# ============ 请求/响应模型 ============

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    image_url: Optional[str] = None
    conversation_id: Optional[str] = Field(
        default=None,
        description="会话标识，用于隔离不同会话的 LangGraph 状态。默认为 'default'。"
    )

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
    height: Optional[float] = Field(None, gt=0, le=300)
    weight: Optional[float] = Field(None, gt=0, le=500)
    age: Optional[int] = Field(None, gt=0, le=150)
    gender: Optional[str] = None
    goal: Optional[str] = None

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


# ============ 健康检查 ============

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/test/stream")
async def test_stream():
    """测试 SSE 流式输出"""
    async def generate():
        text = "这是一段测试文字，用来验证流式输出是否正常工作。"
        for char in text:
            yield f"data: {char}\n\n"
            import asyncio
            await asyncio.sleep(0.03)
        yield f"data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")


# ============ 对话接口（API 变薄，逻辑交给 graph）============

def _build_graph_input(request: ChatRequest, restored_state: dict) -> dict:
    """从请求和恢复状态构造 graph 输入状态

    重要：以下字段是跨轮持久状态，不应被本轮输入重置，必须从 restored_state 读取：
    - pending_confirmation：确认上下文（含 confirmed 标志）
    - requires_confirmation：是否在等待确认

    以下字段是本轮新的输入状态，用请求构造：
    - input_message：用户本轮输入
    - image_info：本轮图片
    - intent / intents：由 classify_intent 在图内重新分类（会覆盖 restored）
    - route_decision：本轮路由决策（重置）
    """
    return {
        # === 本轮输入（不继承）===
        "input_message": request.message,
        "image_info": {
            "has_image": request.image_url is not None,
            "image_url": request.image_url,
        },
        "route_decision": None,
        # === 对话运行时状态（继承）===
        "messages": restored_state.get("messages", []),
        "last_intent": restored_state.get("last_intent"),
        "profile_complete": restored_state.get("profile_complete"),
        "summary_buffer": restored_state.get("summary_buffer", []),
        "turn_count": restored_state.get("turn_count", 0),
        "last_summary_turn": restored_state.get("last_summary_turn", 0),
        "user_profile": restored_state.get("user_profile", {}),
        "daily_stats": restored_state.get("daily_stats", {}),
        # === 跨轮确认状态（必须从 restored_state 恢复，不被本轮重置）===
        # intent/intents 在图内由 classify_intent 重新设置，这里只提供 fallback
        "intent": restored_state.get("intent", "general"),
        "intents": restored_state.get("intents", []),
        "pending_confirmation": restored_state.get("pending_confirmation") or {},
        "requires_confirmation": restored_state.get("requires_confirmation", False),
        # === 分支结果（fan-out 用，本轮结束后图内会写入）===
        "food_branch_result": None,
        "workout_branch_result": None,
        "stats_branch_result": None,
        "food_pending_conf": None,
        "workout_pending_conf": None,
        # === 响应（由图内节点写入）===
        "final_response": None,
        "response": None,
        "food_result": None,
        "workout_result": None,
        "stats_result": None,
        "recipe_result": None,
        # === 兼容旧逻辑（仅 fallback，不主导流程）===
        "pending_stats": None,
        "pending_response": None,
    }


def _post_graph_memory(state: dict, request: ChatRequest, config: dict) -> None:
    """Graph 执行后的记忆处理（side effect，不影响主流程）"""
    try:
        memory_agent.add_conversation_turn(state, request.message, state.get("final_response") or state.get("response", ""))
        memory_agent.extract_and_save_preferences(request.message)
        if memory_agent.should_summarize(state, threshold=10):
            memory_agent.summarize_conversations(state)
        app_obj.update_state(
            config,
            {
                "summary_buffer": state.get("summary_buffer", []),
                "turn_count": state.get("turn_count", 0),
                "last_summary_turn": state.get("last_summary_turn", 0),
            },
        )
    except Exception as e:
        print(f"[Warning] Memory processing failed: {e}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式对话接口 - 核心逻辑交给 LangGraph，API 只负责：
    1. 用 conversation_id 作为 thread_id 恢复 checkpointer 状态
    2. 调用 graph.invoke()（完整执行一次，不重复）
    3. 将节点路由事件 + final_response 流式 SSE 输出
    """
    # 会话隔离：每个 conversation_id 独立一个 thread
    thread_id = request.conversation_id or "default"
    config = {"configurable": {"thread_id": thread_id}}

    # 恢复 checkpointer 状态
    try:
        current_state = app_obj.get_state(config=config)
        if current_state is not None:
            state_data = getattr(current_state, 'values', None) or {}
    except Exception:
        state_data = {}

    graph_input = _build_graph_input(request, state_data)

    async def generate():
        try:
            # 发送会话标识和初始意图
            intent = graph_input.get("intent", "general")
            pending_conf = graph_input.get("pending_confirmation") or {}
            has_pending = bool(pending_conf.get("action"))
            yield f"data: {{\"type\": \"session\", \"thread_id\": \"{thread_id}\"}}\n\n"
            yield f"data: {{\"type\": \"intent\", \"intent\": \"{intent}\"}}\n\n"
            if has_pending:
                action = pending_conf.get("action", "")
                yield f"data: {{\"type\": \"confirm_pending\", \"action\": \"{action}\"}}\n\n"

            # 直接调用 graph.invoke() 完整执行一次，拿到最终状态
            result = app_obj.invoke(graph_input, config=config)

            # 从最终状态提取响应和路由信息
            final_response = (
                result.get("final_response")
                or result.get("response")
                or ""
            )
            committed = result.get("pending_confirmation", {}).get("confirmed")
            route_decision = result.get("route_decision", "")

            # 节点路由事件（体现 graph 执行了哪些节点）
            if route_decision:
                yield f"data: {{\"type\": \"node\", \"node\": \"{route_decision}\"}}\n\n"
            if committed is True:
                yield f"data: {{\"type\": \"commit\", \"result\": \"confirmed\"}}\n\n"
            elif committed is False:
                yield f"data: {{\"type\": \"commit\", \"result\": \"cancelled\"}}\n\n"

            # 流式输出最终文本（SSE）
            if final_response:
                for char in final_response:
                    yield f"data: {{\"type\": \"text\", \"content\": \"{char}\"}}\n\n"
            yield f"data: [DONE]\n\n"

            # Graph 执行后的记忆处理
            import asyncio
            memory_task = asyncio.create_task(_run_post_graph_memory(result, request, config))
            memory_task.add_done_callback(
                lambda t: print(f"[Memory] 记忆处理完成: {t.result()}") if not t.cancelled() and t.exception() is None else print(f"[Memory] 记忆处理失败: {t.exception()}")
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"错误: {str(e)}"
            yield f"data: {{\"type\": \"error\", \"message\": \"{error_msg}\"}}\n\n"
            yield f"data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _run_post_graph_memory(result_state: dict, request: ChatRequest, config: dict) -> str:
    """异步执行记忆处理（不阻塞 SSE 流）"""
    try:
        response_text = (
            result_state.get("final_response")
            or result_state.get("response", "")
        )
        memory_agent.add_conversation_turn(
            result_state, request.message, response_text
        )
        memory_agent.extract_and_save_preferences(request.message)
        if memory_agent.should_summarize(result_state, threshold=10):
            memory_agent.summarize_conversations(result_state)
        app_obj.update_state(
            config,
            {
                "summary_buffer": result_state.get("summary_buffer", []),
                "turn_count": result_state.get("turn_count", 0),
                "last_summary_turn": result_state.get("last_summary_turn", 0),
            },
        )
        return "success"
    except Exception as e:
        print(f"[Warning] Async memory processing failed: {e}")
        return f"error: {e}"


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    同步对话接口 - 逻辑交给 LangGraph。
    API 只负责：接收 → 调用 graph.invoke() → 返回响应。
    """
    thread_id = request.conversation_id or "default"
    config = {"configurable": {"thread_id": thread_id}}

    # 恢复 checkpointer 状态
    try:
        current_state = app_obj.get_state(config=config)
        if current_state is not None:
            state_data = getattr(current_state, 'values', None) or {}
    except Exception:
        state_data = {}

    graph_input = _build_graph_input(request, state_data)

    try:
        # 调用 graph.invoke() 执行完整工作流
        result = app_obj.invoke(graph_input, config=config)

        response_text = result.get("final_response") or result.get("response", "")
        intent = result.get("intent", "general")

        # 记忆处理
        _post_graph_memory(result, request, config)

        return {"response": response_text, "intent": intent, "thread_id": thread_id}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============ 用户档案接口 ============

@app.get("/profile", response_model=ProfileResponse)
async def get_profile():
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
    if not message:
        return {"message": "没有要更新的字段。", "profile": None}

    profile = memory_agent.load_profile()
    if not profile.get("height"):
        # 使用结构化数据创建，跳过 LLM
        result = memory_agent.create_profile_structured(
            height=request.height,
            weight=request.weight,
            age=request.age,
            gender=request.gender,
            goal=request.goal,
        )
        return {"message": "档案创建成功", "profile": result}
    else:
        result = memory_agent.update_profile(message)
        return {"message": "档案更新成功" if result.get("changed") else "未检测到变化", "profile": result}


# ============ 每日统计接口 ============

@app.get("/daily_stats", response_model=DailyStatsResponse)
async def get_daily_stats(date: Optional[str] = None):
    stats = memory_agent.load_daily_stats(date)
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
async def get_conversation_history(conversation_id: Optional[str] = None):
    """从 LangGraph checkpointer 读取对话历史"""
    thread_id = conversation_id or "default"
    config = {"configurable": {"thread_id": thread_id}}
    try:
        current_state = app_obj.get_state(config=config)
        if current_state is not None:
            values = getattr(current_state, 'values', None) or {}
            messages = values.get("messages", [])
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
            return {"history": turns[-20:], "thread_id": thread_id}
    except Exception:
        pass
    return {"history": [], "thread_id": thread_id}


@app.delete("/history")
async def clear_conversation_history(conversation_id: Optional[str] = None):
    """清空对话历史（只清理 messages / summary_buffer，保留 profile_complete）"""
    thread_id = conversation_id or "default"
    config = {"configurable": {"thread_id": thread_id}}
    try:
        current_profile_complete = None
        try:
            state_obj = app_obj.get_state(config=config)
            if state_obj is not None:
                values = getattr(state_obj, 'values', None)
                if values and isinstance(values, dict):
                    current_profile_complete = values.get("profile_complete")
        except Exception:
            pass

        if current_profile_complete is None:
            current_profile_complete = memory_agent.is_profile_complete()

        app_obj.update_state(
            config,
            {
                "messages": [],
                "last_intent": None,
                "profile_complete": current_profile_complete,
                "summary_buffer": [],
                "turn_count": 0,
                "last_summary_turn": 0,
            }
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
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的图片格式: {', '.join(ALLOWED_EXTENSIONS)}")
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"不支持的 MIME 类型")

    contents = await file.read()
    size = len(contents)
    if size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"文件大小超过 {MAX_FILE_SIZE // (1024*1024)}MB")
    if size == 0:
        raise HTTPException(status_code=400, detail="文件为空")

    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = UPLOAD_DIR / unique_name
    with open(file_path, "wb") as f:
        f.write(contents)

    image_url = f"/uploads/images/{unique_name}"
    data_url = f"data:{file.content_type};base64,{base64.b64encode(contents).decode('utf-8')}"

    return UploadResponse(
        success=True,
        filename=unique_name,
        image_url=image_url,
        data_url=data_url,
        content_type=file.content_type,
        size=size
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
