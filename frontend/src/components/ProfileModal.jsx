import { useState } from 'react';
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
      await loadProfile();
      setProfileModalOpen(false);
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