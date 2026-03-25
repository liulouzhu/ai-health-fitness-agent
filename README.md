# AI Health Fitness Agent

一个面向健身与饮食管理场景的智能助手项目。

它不是单纯的聊天机器人，而是一套围绕 `饮食分析 -> 训练建议 -> 食谱推荐 -> 每日统计 -> 个性化记忆` 打通的完整 Agent 系统。项目基于 `LangGraph + FastAPI + React/Vite` 搭建，支持流式对话、图片分析、用户档案、长期记忆、知识检索和前端产品化界面。

## 项目亮点

### 1. 不只是问答，而是完整的健康管理闭环
- 能分析吃了什么，给出营养估算
- 能记录做了什么运动，更新今日消耗
- 能基于剩余额度推荐更适合的食谱
- 能结合用户目标自动计算每日热量和蛋白目标

### 2. 多模态 + 多意图路由
- 支持文本对话
- 支持图片分析食物
- 能识别复合输入，例如“我吃了鸡胸肉，还跑了 30 分钟”
- `food + workout`、`food + stats`、`workout + stats` 这类组合请求会并发处理，响应更自然，链路更完整

### 3. 真正个性化，而不是每次从零开始
- 维护用户档案：身高、体重、年龄、性别、目标
- 持续积累饮食偏好、运动偏好和长期记忆
- 保存每日统计，让推荐和回答更贴近当下状态
- 支持 PostgreSQL 持久化 LangGraph 状态，服务重启后仍可延续上下文

### 4. 检索增强不是摆设
- 食谱推荐和训练建议优先走本地知识库 / 向量检索
- 使用 Qdrant + BM25 混合检索提升召回质量
- 本地检索不足时可回退到 Tavily 做联网补充
- 比完全裸调模型更稳定，也更适合扩展垂直知识

### 5. 不是只有后端，前端已经产品化
- 提供 React + Vite 前端
- 支持 SSE 流式对话体验
- 主聊天区 + 右侧健康面板的信息架构
- 包含今日统计、档案摘要、快捷操作、历史记录、图片上传入口

## 适用场景

- 做一个健身/饮食教练类 AI 产品原型
- 练习 LangGraph 多 Agent 路由与状态管理
- 练习 RAG、用户记忆、日常统计的结合方式
- 作为 AI 健康助手、健康管理 SaaS、垂直智能体的起点项目

## 核心能力

### 智能对话
- `POST /chat/stream` 支持流式输出
- `POST /chat` 支持普通同步响应
- 意图识别后自动路由到对应 Agent

### 食物分析
- 支持文字描述食物营养估算
- 支持图片输入识别食物
- 可将食物计入每日统计

### 运动指导与记录
- 提供训练建议、动作说明、运动消耗估算
- 支持将运动直接计入每日统计

### 食谱推荐
- 结合用户目标、今日剩余热量、剩余蛋白质和偏好进行推荐
- 优先使用本地食谱知识库，必要时联网补充

### 用户记忆
- 用户档案管理
- 偏好提取与增量整合
- 长期记忆摘要
- 对话历史管理

### 图片上传
- 后端提供 `/upload-image`
- 返回本地 `image_url` 和可直接传给多模态模型的 `data_url`
- 方便前端接入本地图片上传能力

## 技术栈

### Backend
- Python
- FastAPI
- LangGraph
- LangChain
- LlamaIndex
- Pydantic v2

### Retrieval / Storage
- Qdrant
- BM25
- PostgreSQL
- Markdown / JSON file memory

### Frontend
- React 18
- Vite 5
- 原生 CSS + CSS Variables
- Fetch API + SSE

## 项目结构

```text
ai_health_fitness_agent/
├── agent/
│   ├── food_agent.py            # 食物分析 Agent
│   ├── graph.py                 # LangGraph 工作流与 checkpointer
│   ├── llm.py                   # LLM 封装
│   ├── memory_agent.py          # 用户档案 / 偏好 / 长期记忆 / 每日统计
│   ├── recipe_agent.py          # 食谱推荐 Agent
│   ├── router_agent.py          # 意图识别、上下文判断、统计查询、确认逻辑
│   ├── state.py                 # AgentState 定义
│   └── workout_agent.py         # 健身指导 Agent
├── frontend/                    # React + Vite 前端
│   ├── src/
│   ├── public/
│   └── package.json
├── knowledge/                   # 知识库与索引构建脚本
├── memory/                      # 用户档案、历史、偏好、每日统计
├── tools/
│   ├── retriever.py             # 混合检索
│   └── search_with_tavily.py    # Tavily 搜索
├── uploads/images/              # 上传图片存储目录
├── api.py                       # FastAPI 应用入口
├── config.py                    # 配置项
├── requirements.txt             # Python 依赖
└── index.html                   # 早期静态原型页（当前推荐使用 frontend/）
```

## 系统架构

### 路由层
`RouterAgent` 负责判断用户当前意图，并结合上下文处理以下情况：
- 一般聊天
- 食物分析 / 食物记录
- 运动建议 / 运动记录
- 食谱推荐
- 今日统计查询
- 用户档案更新
- 对待确认记录进行确认 / 取消

### 记忆层
`MemoryManager` 负责：
- 用户档案读写
- 每日统计管理
- 偏好提取与合并
- 对话历史存储
- 长期记忆摘要

### 上下文层
`ContextManager` 负责：
- 分层上下文装配（system_context / conversation_window / user_memory / task_context / retrieved_knowledge）
- 按 intent 策略选择性注入上下文（不同意图加载不同上下文，避免一股脑塞入）
- 运行时滑动窗口管理（state["messages"]，上限由 `MAX_RECENT_MESSAGES` 控制）
- 统一偏好格式化

**Token 预算机制：**
- `TokenEstimator` 优先使用 `tiktoken`（cl100k_base）真实编码器，失败时自动降级为字符数/3.5 近似估算
- 各 section 独立 token 预算（`MAX_EXTRA_CONTEXT_TOKENS` / `MAX_CONVERSATION_WINDOW_TOKENS` 等），超预算时独立裁剪而非整体截断
- `build_prompt_messages()` 最终输出前输出 debug 日志，格式：`intent=xxx | system=NNN | extra=NNN | conv=NNN | total≈NNN tokens`
- `extra_sections` 参数允许各 agent 注入动态内容（偏好/检索结果等），统一经由 token 预算裁剪

**数据职责边界：**
- `state["messages"]` = LangGraph 运行时滑动窗口（仅保留最近 MAX_RECENT_MESSAGES 条）
- `memory/conversation_history.json` = 完整持久化历史（由 `memory_agent` 在会话结束时写入）
- `memory/longterm_memory.md` = 摘要型长期记忆（不作为原始历史替代品）

### 检索层
- `recipe_agent.py` 使用本地食谱知识库检索
- `workout_agent.py` 使用训练知识库检索
- 当本地检索不足时，可调用 Tavily 搜索增强

### 编排层
`graph.py` 使用 LangGraph 将各节点串成完整工作流，并支持：
- 条件路由
- 多意图并发执行
- PostgreSQL / InMemory checkpointer

## 快速开始

## 1. 安装后端依赖

```bash
pip install -r requirements.txt
```

如果你要使用图片上传接口，建议确认已安装：

```bash
pip install python-multipart
```

## 2. 配置环境变量

复制 `.env.example` 为 `.env`，并填写你的配置：

```env
# DashScope / OpenAI Compatible
LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_API_KEY="your-api-key"
LLM_MODEL="qwen3.5-flash"
VLM_MODEL="qwen3.5-flash"
EMBEDDING_MODEL="text-embedding-v4"

# Optional
TAVILY_API_KEY="your-tavily-api-key"
QDRANT_HOST="localhost"
QDRANT_PORT="6333"
DATABASE_URL="postgresql://postgres:password@localhost:5432/health_agent"
```

## 3. 启动后端

```bash
python api.py
```

启动后：
- API 文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/health`

## 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

默认访问：

```text
http://localhost:5173
```

## 5. 可选依赖服务

### PostgreSQL
用于 LangGraph 状态持久化。

如果未配置 `DATABASE_URL`，系统会自动回退到内存 checkpointer，重启后上下文会丢失。

### Qdrant
用于食谱与训练知识的向量检索。

默认地址：

```text
localhost:6333
```

### Tavily
用于在本地知识检索不足时做联网补充。

## API 概览

| 接口 | 方法 | 说明 |
|---|---|---|
| `/chat/stream` | POST | SSE 流式对话，主聊天入口 |
| `/chat` | POST | 普通同步对话 |
| `/profile` | GET | 获取用户档案 |
| `/profile` | POST | 创建或更新用户档案 |
| `/daily_stats` | GET | 获取今日统计 |
| `/history` | GET | 获取最近对话历史 |
| `/history` | DELETE | 清空对话历史 |
| `/upload-image` | POST | 上传本地图片，返回 `image_url` 和 `data_url` |
| `/health` | GET | 健康检查 |

### `POST /chat/stream`

请求体示例：

```json
{
  "message": "我今天吃了鸡胸肉和米饭，还跑了 30 分钟",
  "image_url": null
}
```

返回为 SSE 数据流：
- 第一段可能返回 `intent`
- 后续逐 token 输出文本
- 结束标记为 `[DONE]`

### `POST /upload-image`

表单上传图片文件，返回：

```json
{
  "success": true,
  "filename": "xxx.png",
  "image_url": "/uploads/images/xxx.png",
  "data_url": "data:image/png;base64,...",
  "content_type": "image/png",
  "size": 12345
}
```

其中 `data_url` 可以直接作为聊天接口中的 `image_url` 传入，便于多模态模型消费。

## 支持的意图

| 意图 | 说明 |
|---|---|
| `food` | 食物识别、营养分析 |
| `food_report` | 用户主动上报已吃食物并计入统计 |
| `workout` | 健身建议、训练指导 |
| `workout_report` | 用户主动上报已做运动并计入统计 |
| `recipe` | 食谱推荐 |
| `stats_query` | 今日统计查询 |
| `profile_update` | 用户档案更新 |
| `confirm` | 对待确认记录进行确认或取消 |
| `general` | 一般对话 |

## 为什么这个项目值得展示

- 它把聊天、记忆、统计、RAG、图片理解和前端体验串成了完整产品链路
- 它不是只有一个“智能回答”接口，而是有真实业务状态
- 它体现了多 Agent 路由和长期记忆在垂直场景中的实际价值
- 它同时具备 Demo 价值、作品集价值和继续演化成产品的可能性

## 后续可扩展方向

- 接入真实食物数据库，降低营养估算误差
- 增加用户登录和多用户隔离
- 新增周期性训练计划与体重趋势追踪
- 增加更细粒度的营养素管理
- 接入可穿戴设备或健康平台数据

## Frontend README

前端的单独说明见：

- [frontend/README.md](frontend/README.md)
