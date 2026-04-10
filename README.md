# AI Health Fitness Agent

一个面向健身与饮食管理场景的智能助手项目。基于 `LangGraph + FastAPI + React/Vite` 搭建，支持流式对话、图片分析、用户档案、长期记忆、知识检索和前端产品化界面。

## 技术栈

### Backend
- Python / FastAPI / LangGraph / LangChain / LlamaIndex / Pydantic v2

### Retrieval / Storage
- Qdrant（向量检索）/ BM25 / PostgreSQL / Markdown / JSON file memory

### Frontend
- React 18 / Vite 5 / 原生 CSS + CSS Variables / Fetch API + SSE

## 项目亮点

- **完整健康管理闭环**：饮食分析 → 训练建议 → 食谱推荐 → 每日统计 → 个性化记忆
- **多模态 + 多意图路由**：支持文本/图片输入，`food + workout`、`food + stats` 等组合请求并发处理
- **真正个性化**：维护用户档案、持续积累偏好、支持 PostgreSQL 持久化 LangGraph 状态
- **检索增强**：Qdrant + BM25 混合检索，必要时回退 Tavily 联网补充
- **多级记忆及统一上下文管理**：用户档案 / 偏好提取 / 长期记忆摘要 / token 预算裁剪 / 分层上下文装配

## 核心功能

| 功能 | 说明 |
|---|---|
| 智能对话 | 意图识别后自动路由，流式/同步两种响应模式 |
| 食物分析 | 文字/图片识别，计入每日营养统计 |
| 运动指导 | 训练建议、动作说明，计入每日消耗 |
| 食谱推荐 | 结合用户目标、剩余额度、偏好进行推荐 |
| 用户记忆 | 档案管理、偏好提取、长期记忆摘要 |
| 图片上传 | `/upload-image` 返回本地 URL 和 Base64 data URL |

## LangGraph 工作流

### 工作流程图

```
check_profile
    │
    ├─ 档案不完整 ──→ general_node ──→ END
    │
    └─ 档案完整 ──→ init_daily_stats ──→ classify_intent
                                                │
                              ┌─────────────────┼─────────────────┐
                              │                 │                 │
                          food_generate     workout_generate   stats_node/recipe/...
                              │                 │                 │
                              ▼                 ▼                 ▼
                         confirm_node ◄── requires_confirmation ─┘
                              │
                              ▼
                         confirm_recovery
                              │
                              ▼
                         commit_node ──→ END
```

### 多意图 Fan-out / Fan-in

当用户输入包含多个意图时，通过 `Send` API 实现真正的并行执行：

```
classify_intent ──fan-out──→ [food_branch, workout_branch]（并发执行）
                                    │            │
                                    └────────────┴──→ multi_join_node
```

### 确认流程（Human-in-the-Loop）

基于 graph state 的 `pending_confirmation` 流转：
1. `food_generate` 设置待确认状态
2. `confirm_node` 展示确认提示
3. 用户下一条消息触发 `confirm_recovery` 读取"是/否"
4. `commit_node` 执行 commit 或取消

### Checkpointer 与状态持久化

- PostgreSQL checkpointer（通过 `DATABASE_URL` 配置）
- 连接失败时自动回退到 `InMemorySaver`
- 每个 `conversation_id` 独立状态，会话完全隔离

## 项目结构

```text
ai_health_fitness_agent/
├── agent/
│   ├── food_agent.py            # 食物分析
│   ├── graph.py                 # LangGraph 工作流定义
│   ├── multi_agent.py           # 多意图 fan-out / fan-in
│   ├── router_agent.py          # 意图分类
│   ├── state.py                 # AgentState 定义
│   ├── memory_agent.py          # 用户档案/偏好/记忆
│   ├── recipe_agent.py          # 食谱推荐
│   ├── workout_agent.py         # 健身指导
│   └── context_manager.py       # 上下文装配与 token 预算
├── frontend/                    # React + Vite 前端
├── knowledge/                   # 知识库与索引构建
├── memory/                      # 用户档案、历史、统计
├── tools/
│   ├── retriever.py             # 混合检索
│   └── search_with_tavily.py   # Tavily 搜索
├── uploads/images/              # 上传图片存储
├── api.py                       # FastAPI 入口
├── config.py                    # 配置项
└── requirements.txt             # Python 依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`：

```env
LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_API_KEY="your-api-key"
LLM_MODEL="qwen3.5-flash"
VLM_MODEL="qwen3.5-flash"
EMBEDDING_MODEL="text-embedding-v4"

# 可选
TAVILY_API_KEY="your-tavily-api-key"
QDRANT_HOST="localhost"
QDRANT_PORT="6333"
DATABASE_URL="postgresql://postgres:password@localhost:5432/health_agent"
```

### 3. 启动后端

```bash
python api.py
```

- API 文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/health`

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173`

### 可选依赖服务

| 服务 | 用途 |
|---|---|
| PostgreSQL | LangGraph 状态持久化 |
| Qdrant | 向量检索 |
| Tavily | 联网补充检索 |

## API 概览

| 接口 | 方法 | 说明 |
|---|---|---|
| `/chat/stream` | POST | SSE 流式对话 |
| `/chat` | POST | 同步对话 |
| `/profile` | GET/POST | 用户档案 |
| `/daily_stats` | GET | 今日统计 |
| `/history` | GET/DELETE | 对话历史 |
| `/upload-image` | POST | 图片上传 |
| `/health` | GET | 健康检查 |

## 支持的意图

`food` / `food_report` / `workout` / `workout_report` / `recipe` / `stats_query` / `profile_update` / `confirm` / `general`

## 适用场景

- 健身/饮食教练类 AI 产品原型
- LangGraph 多 Agent 路由与状态管理实践
- RAG、用户记忆、日常统计结合方式学习
- AI 健康助手 / 健康管理 SaaS 起点项目

## 前端说明

详见 [frontend/README.md](frontend/README.md)
