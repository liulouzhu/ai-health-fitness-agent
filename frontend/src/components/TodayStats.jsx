import { useApp } from '../store/AppContext';
import './TodayStats.css';

const GOAL_LABELS = {
  减脂: '减脂中',
  增肌: '增肌中',
  维持: '维持中',
  cut: '减脂中',
  bulk: '增肌中',
  maintain: '维持中',
};

export default function TodayStats() {
  const { state } = useApp();
  const { stats, statsLoading } = state;

  if (statsLoading) {
    return (
      <div className="today-stats">
        <div className="stats-header">
          <span className="stats-title">今日概览</span>
          <span className="stats-date">{new Date().toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })}</span>
        </div>
        <div className="stats-skeleton">
          {[...Array(4)].map((_, i) => <div key={i} className="stats-skeleton-item" />)}
        </div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="today-stats">
        <div className="stats-header">
          <span className="stats-title">今日概览</span>
        </div>
        <div className="stats-empty">暂无数据</div>
      </div>
    );
  }

  const { consumed_calories, remaining_calories, consumed_protein, remaining_protein, burned_calories, date } = stats;

  // Calculate percentages for progress bars
  const targetCal = consumed_calories + remaining_calories;
  const targetPro = consumed_protein + remaining_protein;
  const calPct = targetCal > 0 ? Math.min(100, Math.round((consumed_calories / targetCal) * 100)) : 0;
  const proPct = targetPro > 0 ? Math.min(100, Math.round((consumed_protein / targetPro) * 100)) : 0;

  const dateDisplay = date
    ? new Date(date).toLocaleDateString('zh-CN', { month: 'short', day: 'numeric', weekday: 'short' })
    : new Date().toLocaleDateString('zh-CN', { month: 'short', day: 'numeric', weekday: 'short' });

  return (
    <div className="today-stats">
      <div className="stats-header">
        <span className="stats-title">今日概览</span>
        <span className="stats-date">{dateDisplay}</span>
      </div>

      <div className="stats-calories">
        <div className="stats-cal-row">
          <div className="stats-cal-item">
            <span className="stats-cal-label">已摄入</span>
            <span className="stats-cal-value consumed">{consumed_calories}</span>
            <span className="stats-cal-unit">kcal</span>
          </div>
          <div className="stats-cal-divider" />
          <div className="stats-cal-item">
            <span className="stats-cal-label">剩余</span>
            <span className="stats-cal-value remaining">{remaining_calories}</span>
            <span className="stats-cal-unit">kcal</span>
          </div>
        </div>
        <div className="stats-progress">
          <div className="stats-progress-bar">
            <div
              className="stats-progress-fill consumed"
              style={{ width: `${calPct}%` }}
            />
          </div>
          <span className="stats-progress-label">{calPct}%</span>
        </div>
      </div>

      <div className="stats-row">
        <div className="stats-mini-card">
          <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M10 2C7 2 5 5 5 8c0 4 5 10 5 10s5-6 5-10c0-3-2-6-5-6Z" strokeLinecap="round"/>
            <circle cx="10" cy="8" r="1.5"/>
          </svg>
          <div className="stats-mini-content">
            <span className="stats-mini-value">{consumed_protein.toFixed(0)}</span>
            <span className="stats-mini-label">蛋白质 g</span>
          </div>
        </div>

        <div className="stats-mini-card">
          <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="10" cy="10" r="7"/>
            <path d="M10 6v4l3 3" strokeLinecap="round"/>
          </svg>
          <div className="stats-mini-content">
            <span className="stats-mini-value consumed">{burned_calories}</span>
            <span className="stats-mini-label">运动消耗</span>
          </div>
        </div>
      </div>

    </div>
  );
}
