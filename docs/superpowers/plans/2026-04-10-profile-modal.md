# 用户档案弹窗实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现用户档案弹窗功能：档案为空时自动弹出，填写后保存；之后可随时点击编辑。

**Architecture:** 新建 `ProfileModal` 组件作为模态弹窗；在 `AppContext` 添加弹窗状态管理；在 `Sidebar` 空档案时触发弹窗；在 `ProfileCard` 空白区域添加点击触发编辑。

**Tech Stack:** React (JSX), CSS Modules pattern (using .css files), Fetch API

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `frontend/src/services/api.js` | 添加 `saveProfile` API |
| `frontend/src/store/AppContext.jsx` | 添加 `profileModalOpen` 状态和 `setProfileModalOpen` 方法 |
| `frontend/src/components/ProfileModal.jsx` | 新建 — 档案表单弹窗组件 |
| `frontend/src/components/ProfileModal.css` | 新建 — 弹窗样式 |
| `frontend/src/components/ProfileCard.jsx` | 修改 — 空白状态添加点击事件 |
| `frontend/src/components/Sidebar.jsx` | 修改 — 空档案时渲染弹窗 |

---

## Task 1: 添加 saveProfile API

**Files:**
- Modify: `frontend/src/services/api.js`

- [ ] **Step 1: 在 api.js 添加 saveProfile 函数**

在 `fetchProfile` 函数后添加：

```javascript
/**
 * 保存用户档案（创建或更新）
 * @param {Object} profileData - { height, weight, age, gender, goal }
 * @returns {Promise<Object>}
 */
export async function saveProfile(profileData) {
  const res = await fetch(`${API_BASE}profile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(profileData),
  });
  if (!res.ok) throw new Error('保存档案失败');
  return res.json();
}
```

- [ ] **Step 2: 提交**

```bash
git add frontend/src/services/api.js
git commit -m "feat: add saveProfile API"
```

---

## Task 2: AppContext 添加弹窗状态

**Files:**
- Modify: `frontend/src/store/AppContext.jsx`

- [ ] **Step 1: 在 initialState 添加 profileModalOpen**

```javascript
const initialState = {
  // ...existing fields...
  activeView: 'chat',
  sidebarOpen: true,
  profileModalOpen: false,  // 新增
};
```

- [ ] **Step 2: 在 reducer 添加 SET_PROFILE_MODAL_OPEN case**

```javascript
case 'SET_PROFILE_MODAL_OPEN':
  return { ...state, profileModalOpen: action.payload };
```

- [ ] **Step 3: 在 AppProvider 的 value 中暴露 setProfileModalOpen**

在 `return` 部分的 `AppContext.Provider value` 中添加：

```javascript
setProfileModalOpen: (open) => dispatch({ type: 'SET_PROFILE_MODAL_OPEN', payload: open }),
```

- [ ] **Step 4: 提交**

```bash
git add frontend/src/store/AppContext.jsx
git commit -m "feat: add profileModalOpen state to AppContext"
```

---

## Task 3: 创建 ProfileModal 组件

**Files:**
- Create: `frontend/src/components/ProfileModal.jsx`
- Create: `frontend/src/components/ProfileModal.css`

- [ ] **Step 1: 创建 ProfileModal.jsx**

```jsx
import { useState, useEffect } from 'react';
import { useApp } from '../store/AppContext';
import { saveProfile } from '../services/api';
import './ProfileModal.css';

const GENDER_OPTIONS = [
  { value: 'male', label: '男' },
  { value: 'female', label: '女' },
];

const GOAL_OPTIONS = [
  { value: '减脂', label: '减脂' },
  { value: '增肌', label: '增肌' },
  { value: '维持', label: '维持' },
];

export default function ProfileModal({ isInitial = false }) {
  const { state, dispatch, loadProfile, setProfileModalOpen } = useApp();
  const [form, setForm] = useState({
    height: '',
    weight: '',
    age: '',
    gender: 'male',
    goal: '维持',
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const { height, weight, age } = form;
    if (!height || !weight || !age) {
      setError('请填写身高、体重和年龄');
      return;
    }

    setSaving(true);
    setError('');

    try {
      await saveProfile({
        height: parseFloat(height),
        weight: parseFloat(weight),
        age: parseInt(age, 10),
        gender: form.gender,
        goal: form.goal,
      });

      // 重新加载档案
      await loadProfile();

      // 关闭弹窗
      setProfileModalOpen(false);

      // 如果是首次创建，设置 localStorage 标记
      if (isInitial) {
        localStorage.setItem('profileModalShown', 'true');
      }
    } catch (err) {
      setError(err.message || '保存失败，请重试');
    } finally {
      setSaving(false);
    }
  };

  const handleClose = () => {
    setProfileModalOpen(false);
    if (isInitial) {
      localStorage.setItem('profileModalShown', 'true');
    }
  };

  const handleBackdropClick = (e) => {
    if (e.target === e.currentTarget) {
      handleClose();
    }
  };

  return (
    <div className="modal-backdrop" onClick={handleBackdropClick}>
      <div className="modal" role="dialog" aria-modal="true">
        <div className="modal-header">
          <h2 className="modal-title">{isInitial ? '欢迎使用健身助手' : '编辑用户档案'}</h2>
          <button className="modal-close" onClick={handleClose} aria-label="关闭">
            <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M5 5l10 10M15 5L5 15" strokeLinecap="round"/>
            </svg>
          </button>
        </div>

        {isInitial && (
          <div className="modal-welcome">
            请填写你的基本信息，我会为你制定个性化的健身计划。
          </div>
        )}

        <form className="modal-form" onSubmit={handleSubmit}>
          <div className="form-row">
            <div className="form-field">
              <label className="form-label" htmlFor="height">身高 (cm)</label>
              <input
                id="height"
                type="number"
                className="form-input"
                placeholder="例如：170"
                value={form.height}
                onChange={(e) => handleChange('height', e.target.value)}
                min="100"
                max="250"
              />
            </div>
            <div className="form-field">
              <label className="form-label" htmlFor="weight">体重 (kg)</label>
              <input
                id="weight"
                type="number"
                className="form-input"
                placeholder="例如：65"
                value={form.weight}
                onChange={(e) => handleChange('weight', e.target.value)}
                min="30"
                max="200"
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-field">
              <label className="form-label" htmlFor="age">年龄 (岁)</label>
              <input
                id="age"
                type="number"
                className="form-input"
                placeholder="例如：25"
                value={form.age}
                onChange={(e) => handleChange('age', e.target.value)}
                min="10"
                max="100"
              />
            </div>
            <div className="form-field">
              <label className="form-label" htmlFor="gender">性别</label>
              <select
                id="gender"
                className="form-select"
                value={form.gender}
                onChange={(e) => handleChange('gender', e.target.value)}
              >
                {GENDER_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="form-field">
            <label className="form-label" htmlFor="goal">健身目标</label>
            <select
              id="goal"
              className="form-select"
              value={form.goal}
              onChange={(e) => handleChange('goal', e.target.value)}
            >
              {GOAL_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          {error && <div className="form-error">{error}</div>}

          <div className="modal-actions">
            <button type="button" className="btn-secondary" onClick={handleClose} disabled={saving}>
              取消
            </button>
            <button type="submit" className="btn-primary" disabled={saving}>
              {saving ? '保存中...' : '保存'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 创建 ProfileModal.css**

```css
.modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  animation: fadeIn var(--transition-fast);
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.modal {
  background: var(--color-bg-card);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-lg);
  width: 420px;
  max-width: calc(100vw - 32px);
  max-height: calc(100vh - 64px);
  overflow-y: auto;
  animation: slideUp var(--transition-base);
}

@keyframes slideUp {
  from { transform: translateY(16px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-5) var(--space-6);
  border-bottom: 1px solid var(--color-border-light);
}

.modal-title {
  font-size: var(--text-lg);
  font-weight: 600;
  color: var(--color-text-primary);
}

.modal-close {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius-sm);
  color: var(--color-text-tertiary);
  transition: background var(--transition-fast), color var(--transition-fast);
}

.modal-close:hover {
  background: var(--color-bg-secondary);
  color: var(--color-text-primary);
}

.modal-close svg {
  width: 18px;
  height: 18px;
}

.modal-welcome {
  padding: var(--space-4) var(--space-6);
  background: var(--color-accent-bg);
  color: var(--color-text-secondary);
  font-size: var(--text-sm);
  line-height: 1.5;
}

.modal-form {
  padding: var(--space-6);
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.form-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-4);
}

.form-field {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.form-label {
  font-size: var(--text-xs);
  font-weight: 500;
  color: var(--color-text-tertiary);
}

.form-input,
.form-select {
  height: 40px;
  padding: 0 var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-bg);
  font-size: var(--text-sm);
  color: var(--color-text-primary);
  transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
}

.form-input:focus,
.form-select:focus {
  outline: none;
  border-color: var(--color-accent);
  box-shadow: 0 0 0 3px var(--color-accent-bg);
}

.form-input::placeholder {
  color: var(--color-text-tertiary);
}

.form-select {
  cursor: pointer;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath d='M3 4.5l3 3 3-3' stroke='%239E9E9E' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 12px center;
  padding-right: 32px;
}

.form-error {
  padding: var(--space-3);
  background: rgba(201, 107, 107, 0.08);
  border: 1px solid rgba(201, 107, 107, 0.2);
  border-radius: var(--radius-md);
  color: var(--color-error);
  font-size: var(--text-sm);
}

.modal-actions {
  display: flex;
  gap: var(--space-3);
  justify-content: flex-end;
  padding-top: var(--space-2);
}

.btn-primary,
.btn-secondary {
  height: 40px;
  padding: 0 var(--space-5);
  border-radius: var(--radius-md);
  font-size: var(--text-sm);
  font-weight: 500;
  transition: background var(--transition-fast), opacity var(--transition-fast);
}

.btn-primary {
  background: var(--color-accent);
  color: var(--color-text-inverse);
}

.btn-primary:hover:not(:disabled) {
  background: var(--color-accent-light);
}

.btn-primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-secondary {
  background: var(--color-bg-secondary);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
}

.btn-secondary:hover:not(:disabled) {
  background: var(--color-border-light);
  color: var(--color-text-primary);
}

.btn-secondary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
```

- [ ] **Step 3: 提交**

```bash
git add frontend/src/components/ProfileModal.jsx frontend/src/components/ProfileModal.css
git commit -m "feat: add ProfileModal component"
```

---

## Task 4: ProfileCard 添加点击触发编辑

**Files:**
- Modify: `frontend/src/components/ProfileCard.jsx`
- Modify: `frontend/src/components/ProfileCard.css`

- [ ] **Step 1: 修改 ProfileCard.jsx — 空白状态添加点击和编辑按钮**

在 `ProfileCard.jsx` 中：

导入 useApp 后添加 `setProfileModalOpen`:

```jsx
const { state, setProfileModalOpen } = useApp();
```

修改空状态返回部分，将整个 `profile-empty` div 包裹为可点击：

```jsx
if (!profile) {
  return (
    <div className="profile-card">
      <div className="profile-card-header">
        <span className="profile-card-title">用户档案</span>
      </div>
      <div
        className="profile-empty"
        onClick={() => setProfileModalOpen(true)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && setProfileModalOpen(true)}
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <circle cx="12" cy="8" r="4"/>
          <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" strokeLinecap="round"/>
        </svg>
        <span>还没有档案</span>
        <p>点击填写你的基本信息</p>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 修改 ProfileCard.css — 添加 cursor: pointer**

在 `.profile-empty` 样式中添加：

```css
.profile-empty {
  cursor: pointer;
  /* ...existing styles... */
}
```

- [ ] **Step 3: 提交**

```bash
git add frontend/src/components/ProfileCard.jsx frontend/src/components/ProfileCard.css
git commit -m "feat: ProfileCard empty state clickable to open modal"
```

---

## Task 5: Sidebar 空档案时渲染弹窗

**Files:**
- Modify: `frontend/src/components/Sidebar.jsx`

- [ ] **Step 1: 修改 Sidebar.jsx — 导入并渲染 ProfileModal**

```jsx
import ProfileModal from './ProfileModal';

export default function Sidebar() {
  const { loadStats, loadProfile, state, setProfileModalOpen } = useApp();

  useEffect(() => {
    if (state.backendAlive) {
      loadStats();
      loadProfile();
    }
  }, [state.backendAlive, loadStats, loadProfile]);

  // 空档案且从未弹过弹窗时，自动弹出
  useEffect(() => {
    if (state.profile === null && !state.profileLoading) {
      const hasShown = localStorage.getItem('profileModalShown');
      if (!hasShown) {
        setProfileModalOpen(true);
      }
    }
  }, [state.profile, state.profileLoading, setProfileModalOpen]);

  return (
    <div className="sidebar">
      <div className="sidebar-section">
        <TodayStats />
      </div>
      <div className="sidebar-divider" />
      <div className="sidebar-section">
        <ProfileCard />
      </div>
      <div className="sidebar-divider" />
      <div className="sidebar-section">
        <QuickActions />
      </div>

      {/* 空档案时渲染首次弹窗 */}
      {state.profile === null && (
        <ProfileModal isInitial={true} />
      )}
    </div>
  );
}
```

- [ ] **Step 2: 提交**

```bash
git add frontend/src/components/Sidebar.jsx
git commit -m "feat: Sidebar renders ProfileModal when profile is empty"
```

---

## Task 6: 整体测试

- [ ] **Step 1: 启动前端**

```bash
cd frontend && npm run dev
```

- [ ] **Step 2: 验证功能**

1. 访问页面，检查是否自动弹出档案填写弹窗
2. 填写表单后点击保存，验证弹窗关闭且档案显示正确
3. 点击侧边栏档案空白区域，验证编辑弹窗打开
4. 刷新页面，验证不再自动弹出（localStorage 标记生效）
5. 点击档案编辑，验证弹窗可正常编辑保存

---

## 实施检查清单

- [ ] Task 1: saveProfile API
- [ ] Task 2: AppContext profileModalOpen 状态
- [ ] Task 3: ProfileModal 组件和样式
- [ ] Task 4: ProfileCard 点击触发
- [ ] Task 5: Sidebar 空档案渲染弹窗
- [ ] Task 6: 整体测试