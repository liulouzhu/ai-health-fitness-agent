# 健身健康助手 - 前端

## 技术栈

- **React 18** + **Vite 5**
- 纯 CSS（CSS Variables），无 UI 组件库
- Context + useReducer 轻量状态管理
- Fetch API + SSE 流式渲染

## 快速启动

```bash
cd frontend
npm install

# 开发模式（访问 http://localhost:5173）
npm run dev

# 生产构建
npm run build
```

## 目录结构

```
frontend/
├── public/
│   └── favicon.svg
├── src/
│   ├── components/          # 可复用 UI 组件
│   │   ├── Header.jsx          # 顶部导航栏
│   │   ├── Sidebar.jsx         # 右侧边栏容器
│   │   ├── TodayStats.jsx      # 今日概览（热量/蛋白质/运动）
│   │   ├── ProfileCard.jsx     # 用户档案卡片
│   │   ├── QuickActions.jsx     # 快捷问题按钮
│   │   ├── ChatMessage.jsx      # 聊天消息气泡
│   │   └── BackendErrorBanner.jsx  # 后端未连接提示
│   ├── views/               # 页面级视图
│   │   ├── ChatPanel.jsx    # 主聊天视图
│   │   └── HistoryView.jsx  # 历史记录视图
│   ├── services/             # API 层
│   │   └── api.js           # REST + SSE 封装
│   ├── store/                # 状态管理
│   │   └── AppContext.jsx   # 全局 Context + Reducer
│   ├── styles/               # 全局样式
│   │   ├── variables.css    # CSS 变量（颜色/间距/字体）
│   │   └── global.css       # 全局 reset + 基础样式
│   ├── App.jsx
│   ├── App.css
│   └── main.jsx
├── index.html
├── package.json
├── vite.config.js
└── .gitignore
```

## 设计说明

### 视觉风格
- **暖白/米白背景**，低饱和自然绿强调色（鼠尾草绿 #7BA05B）
- 深灰主文字（#2C2C2C），非纯黑
- 卡片 + 轻阴影 + 圆角营造层次感
- 克制使用渐变、模糊、留白

### 布局
- **桌面端**：左侧主聊天区（flex 主区域）+ 右侧健康面板（固定宽度）
- **移动端**：纵向堆叠，聊天区填满视口，右侧栏最大 40dvh
- 聊天消息区独立滚动，输入框始终固定在底部
- Header 导航切换「对话」和「历史」两个主视图

### 聊天功能
- SSE 流式渲染，实时逐 token 输出
- Intent 元数据标签（食物 / 运动 / 食谱 / 统计等）
- 用户消息靠右，助手消息靠左，气泡各有方向对应的小尖角
- 消息气泡支持基础 Markdown 渲染（加粗、列表、分割线）
- 加载态 / 错误态 / 空状态 完整处理
- 图片 URL 输入入口（可折叠）

### 右侧健康面板
- **今日概览**：已摄入热量 + 进度条 + 剩余热量、蛋白质克数、运动消耗
- **用户档案**：身高/体重/年龄/性别/目标/每日目标
- **快捷问题**：点击自动填入输入框

### 错误处理
- 后端未启动：顶部横幅提示
- 接口 404/500：对应空状态 / 错误文案
- SSE 流中断：显示错误消息

## API 适配

严格按现有后端接口：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/chat/stream` | POST | SSE 流式对话（第一段 intent，后续逐 token） |
| `/chat` | POST | 普通对话 |
| `/profile` | GET/POST | 用户档案 |
| `/daily_stats` | GET | 今日统计 |
| `/history` | GET/DELETE | 对话历史 |
| `/upload-image` | POST | 图片上传 |

## 环境要求

- Node.js >= 16
- 后端 FastAPI 运行在 `http://localhost:8000`
- Vite 开发服务器通过 proxy 代理 API 请求，无需额外配置 CORS
