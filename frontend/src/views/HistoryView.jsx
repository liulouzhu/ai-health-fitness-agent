import { useEffect } from 'react';
import { useApp } from '../store/AppContext';
import './HistoryView.css';

export default function HistoryView() {
  const { state, loadHistory } = useApp();
  const { history, historyLoading, historyError } = state;

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  const handleClear = async () => {
    if (!window.confirm('确定要清除所有对话历史吗？')) return;
    const { clearHistory } = await import('../services/api');
    await clearHistory();
    loadHistory();
  };

  if (historyLoading) {
    return (
      <div className="history-view">
        <div className="history-header">
          <h2 className="history-title">对话历史</h2>
        </div>
        <div className="history-skeleton">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="history-skeleton-item">
              <div className="history-skeleton-time" />
              <div className="history-skeleton-content" />
              <div className="history-skeleton-content short" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="history-view">
      <div className="history-header">
        <h2 className="history-title">对话历史</h2>
        {history.length > 0 && (
          <button className="history-clear-btn" onClick={handleClear}>
            <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M5 5l10 10M15 5L5 15" strokeLinecap="round"/>
            </svg>
            清除
          </button>
        )}
      </div>

      {historyError && (
        <div className="history-error">
          <span>{historyError}</span>
        </div>
      )}

      {history.length === 0 ? (
        <div className="history-empty">
          <svg viewBox="0 0 48 48" fill="none">
            <circle cx="24" cy="24" r="20" fill="var(--color-accent)" opacity="0.08"/>
            <path d="M16 20h16M16 24h12M16 28h8" stroke="var(--color-accent)" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <p>暂无对话历史</p>
        </div>
      ) : (
        <div className="history-list">
          {history.map((item, index) => (
            <HistoryItem key={index} item={item} />
          ))}
        </div>
      )}
    </div>
  );
}

function HistoryItem({ item }) {
  const { timestamp, user, agent } = item;

  const timeDisplay = timestamp
    ? new Date(timestamp).toLocaleString('zh-CN', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    : '';

  return (
    <div className="history-item">
      <div className="history-item-time">{timeDisplay}</div>
      <div className="history-item-exchange">
        <div className="history-item-message user">
          <div className="history-item-role">你</div>
          <div className="history-item-text">{user}</div>
        </div>
        <div className="history-item-message assistant">
          <div className="history-item-role">助手</div>
          <div className="history-item-text">{agent}</div>
        </div>
      </div>
    </div>
  );
}
