import { useApp } from '../store/AppContext';
import './ProfileCard.css';

const GOAL_MAP = {
  减脂: '减脂',
  增肌: '增肌',
  维持: '维持',
  cut: '减脂',
  bulk: '增肌',
  maintain: '维持',
};

const GENDER_MAP = {
  male: '男',
  female: '女',
  男: '男',
  女: '女',
};

export default function ProfileCard() {
  const { state } = useApp();
  const { profile, profileLoading } = state;

  if (profileLoading) {
    return (
      <div className="profile-card">
        <div className="profile-card-header">
          <span className="profile-card-title">用户档案</span>
        </div>
        <div className="profile-skeleton">
          {[...Array(4)].map((_, i) => <div key={i} className="profile-skeleton-item" />)}
        </div>
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="profile-card">
        <div className="profile-card-header">
          <span className="profile-card-title">用户档案</span>
        </div>
        <div className="profile-empty">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="12" cy="8" r="4"/>
            <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" strokeLinecap="round"/>
          </svg>
          <span>还没有档案</span>
          <p>在对话框中输入你的基本信息，即可创建档案</p>
        </div>
      </div>
    );
  }

  const goalDisplay = GOAL_MAP[profile.goal] || profile.goal || '-';
  const genderDisplay = GENDER_MAP[profile.gender] || profile.gender || '-';

  const items = [
    { label: '身高', value: `${profile.height} cm`, icon: '↕' },
    { label: '体重', value: `${profile.weight} kg`, icon: '⚖' },
    { label: '年龄', value: `${profile.age} 岁`, icon: '⏱' },
    { label: '性别', value: genderDisplay, icon: '♀' },
  ];

  return (
    <div className="profile-card">
      <div className="profile-card-header">
        <span className="profile-card-title">用户档案</span>
        <span className={`profile-goal-badge ${goalDisplay}`}>{goalDisplay}</span>
      </div>

      <div className="profile-items">
        {items.map((item) => (
          <div key={item.label} className="profile-item">
            <span className="profile-item-label">{item.label}</span>
            <span className="profile-item-value">{item.value}</span>
          </div>
        ))}
      </div>

      <div className="profile-targets">
        <div className="profile-target-item">
          <span className="profile-target-value">{profile.target_calories}</span>
          <span className="profile-target-label">目标热量 kcal</span>
        </div>
        <div className="profile-target-item">
          <span className="profile-target-value">{profile.target_protein}</span>
          <span className="profile-target-label">目标蛋白 g</span>
        </div>
      </div>
    </div>
  );
}
