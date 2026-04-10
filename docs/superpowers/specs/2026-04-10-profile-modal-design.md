# 用户档案弹窗设计

## 概述

档案为空时自动弹出模态弹窗引导用户填写；之后用户可随时点击侧边栏"用户档案"区域打开编辑弹窗。

## 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| 身高 | number (cm) | 输入框，必填 |
| 体重 | number (kg) | 输入框，必填 |
| 年龄 | number (岁) | 输入框，必填 |
| 性别 | select | 男 / 女，必填 |
| 目标 | select | 减脂 / 增肌 / 维持，必填 |

目标热量和目标蛋白由后端 `_calculate_targets` 自动计算。

## 交互

- 点击"保存" → 调用 `POST /profile` API → 成功后关闭弹窗，刷新档案显示
- 点击"取消"或遮罩层 → 关闭弹窗
- 新建模式：弹窗顶部显示欢迎提示语
- 编辑模式（点击档案区域）：无提示语，直接显示表单，可修改任意字段

## 触发规则

- **自动触发**：档案为空（`profile === null`）且尚未弹过（localStorage 标记）时弹出
- **手动触发**：用户点击 `ProfileCard` 的空白状态区域或编辑图标，弹出编辑弹窗

## 文件变更

| 操作 | 文件 |
|------|------|
| 新增 | `frontend/src/components/ProfileModal.jsx` |
| 新增 | `frontend/src/components/ProfileModal.css` |
| 修改 | `frontend/src/components/ProfileCard.jsx` — 添加点击触发编辑弹窗 |
| 修改 | `frontend/src/components/Sidebar.jsx` — 空档案时渲染弹窗 |
| 修改 | `frontend/src/store/AppContext.jsx` — 添加 `profileModalOpen` 状态和 `setProfileModalOpen` |
| 修改 | `frontend/src/services/api.js` — 添加 `saveProfile` API 调用 |