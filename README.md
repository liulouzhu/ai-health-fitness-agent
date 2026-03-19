# 健身健康智能助手

基于 LangGraph 的健身健康智能体，提供食物营养分析、食谱推荐、健身指导等功能。

## 使用技术
- **Python**
- **Langgraph**
- **LlamaIndex**
- **Fastapi**
- **RAG**

## ToDo-List
- 修一下某些bug
- 完善前端界面

## 功能特性

- **食物识别与营养分析**：上传食物图片，自动识别并计算卡路里和营养成分
- **食谱推荐**：根据用户档案和每日摄入情况，智能推荐合适的食谱
- **健身指导**：提供训练计划、动作要领、热量消耗统计
- **用户档案管理**：记录身高、体重、年龄、性别、健身目标，自动计算每日目标
- **每日统计**：追踪摄入热量、蛋白质、运动消耗，计算剩余额度

## 项目结构

```
ai_health_fitness_agent/
├── api.py                 # FastAPI 服务入口
├── index.html             # Web 聊天界面
├── config.py              # 配置文件
├── agent/
│   ├── graph.py          # LangGraph 工作流定义
│   ├── state.py          # 状态类型定义
│   ├── router_agent.py   # 意图分类与路由
│   ├── food_agent.py     # 食物分析 Agent
│   ├── workout_agent.py  # 健身指导 Agent
│   ├── recipe_agent.py   # 食谱推荐 Agent
│   ├── memory_agent.py   # 用户记忆管理
│   └── llm.py            # LLM 调用封装
├── tools/
│   └── search_with_tavily.py  # Tavily 搜索工具
├── knowledge/
│   ├── nutrition.md       # 营养知识库
│   ├── fitness-form-guide.md  # 健身动作指南
│   ├── fitness_recipes.md # 食谱知识库
│   └── build_*.py        # 向量索引构建脚本
└── memory/
    ├── memory.md         # 用户档案存储
    └── daily_stats/      # 每日统计存储
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
LLM_BASE_URL=https://api.example.com/v1
LLM_API_KEY=your_api_key
LLM_MODEL=claude-3-sonnet
VLM_MODEL=claude-3-sonnet
EMBEDDING_MODEL=text-embedding-3-small
TAVILY_API_KEY=your_tavily_key
```

### 3. 启动 API 服务

```bash
python api.py
```

服务启动后访问 http://localhost:8000 查看 API 文档。

### 4. 启动 Web 界面

直接在浏览器中打开 `index.html` 文件。

## API 接口

### POST /chat

对话接口，支持文本和图片。

```json
{
  "message": "推荐一个食谱",
  "image_url": null
}
```

响应：

```json
{
  "response": "【推荐食谱组合方案】...",
  "intent": "recipe"
}
```

### GET /profile

获取用户档案。

### POST /profile

创建或更新用户档案。

```json
{
  "height": 175,
  "weight": 70,
  "age": 25,
  "gender": "male",
  "goal": "cut"
}
```

### GET /daily_stats

获取今日统计。

### GET /health

健康检查。

## 意图分类

系统支持以下意图分类：

| 意图 | 说明 |
|------|------|
| `food` | 食物识别、营养分析 |
| `workout` | 健身计划、运动建议 |
| `recipe` | 食谱推荐 |
| `stats_query` | 统计查询 |
| `profile_update` | 档案更新 |
| `confirm` | 确认操作 |
| `general` | 一般对话 |

## 上下文理解

系统支持上下文相关的对话续承：

- 用户说"推荐一个食谱" → 路由到 `recipe`
- 用户说"换一个" → 基于上一轮意图，识别为对食谱的延续请求

## 调试工具

- `test_checkpoint.py`：测试 InMemorySaver 状态持久化
- 运行 `python test_checkpoint.py` 查看 checkpoint 存储状态

## 依赖服务

- **Qdrant**（可选）：本地向量数据库，用于食谱检索
  - 地址：`localhost:6333`
  - 集合名：`recipes`
- **Tavily API**（可选）：网络搜索增强
