# 健身健康助手 - 前端

## 技术栈

- **React 18** + **Vite 5**
- 纯 CSS（CSS Variables），无 UI 组件库
- Context + useReducer 轻量状态管理
- Fetch API + SSE 流式渲染

## 目录结构

```
frontend/
├── public/
│   └── favicon.svg
├── src/
│   ├── components/          # 可复用 UI 组件
│   │   ├── Header.jsx       # 顶部导航栏
│   │   ├── Sidebar.jsx      # 右侧边栏容器
│   │   ├── TodayStats.jsx   # 今日统计面板
│   │   ├── ProfileCard.jsx   # 用户档案卡片
│   │   ├── QuickActions.jsx  # 快捷问题按钮
│   │   ├── ChatMessage.jsx  # 聊天消息气泡
│   │   └── BackendErrorBanner.jsx  # 后端未启动提示
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
└── vite.config.js
```

## 启动方式

```bash
# 安装依赖
cd frontend
npm install

# 开发模式（默认请求 http://localhost:8000）
npm run dev

# 生产构建
npm run build
```

## 设计说明

### 视觉风格
- **暖白/米白背景**，低饱和自然绿强调色（鼠尾草绿 #7BA05B）
- 深灰主文字（#2C2C2C），非纯黑
- 卡片 + 轻阴影 + 圆角营造层次感
- 克制使用渐变、模糊、留白

### 布局
- **桌面端**：左侧主聊天区 + 右侧健康面板（统计 / 档案 / 快捷操作）
- **移动端**：纵向堆叠，聊天区优先
- Header 导航切换「对话」和「历史」两个主视图

### 聊天功能
- SSE 流式渲染，实时逐 token 输出
- Intent 元数据标签（食物 / 运动 / 食谱 / 统计等）
- 消息气泡支持基础 Markdown 渲染
- 加载态 / 错误态 / 空状态 完整处理
- 图片 URL 输入入口（可折叠）

### 数据处理
- `/chat/stream` SSE 流式
- `/history` 返回 `timestamp / user / agent` 结构，正确渲染对话历史
- `/daily_stats` 热量 + 蛋白质环形进度条
- `/profile` 空档案时显示引导文案

### 错误处理
- 后端未启动：顶部横幅提示
- 接口 404/500：对应空状态 / 错误文案
- SSE 流中断：显示错误消息

## API 适配

严格按现有后端接口：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/chat/stream` | POST | SSE 流式对话 |
| `/chat` | POST | 普通对话 |
| `/profile` | GET/POST | 用户档案 |
| `/daily_stats` | GET | 今日统计 |
| `/history` | GET/DELETE | 对话历史 |
