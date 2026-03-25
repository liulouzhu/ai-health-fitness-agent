import './QuickActions.css';

const ACTIONS = [
  {
    label: '分析我今天吃的东西',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M3 8a5 5 0 0 1 10 0v1a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V8Z" strokeLinecap="round"/>
        <path d="M8 5v6M5 8h6" strokeLinecap="round"/>
      </svg>
    ),
  },
  {
    label: '推荐高蛋白菜谱',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M10 2v4M10 14v4M2 10h4M14 10h4" strokeLinecap="round"/>
        <circle cx="10" cy="10" r="3"/>
      </svg>
    ),
  },
  {
    label: '给我安排今天训练',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M5 10h10M10 5v10" strokeLinecap="round"/>
        <rect x="3" y="3" width="14" height="14" rx="3"/>
      </svg>
    ),
  },
  {
    label: '看看我今天还剩多少热量',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M3 8a5 5 0 0 1 10 0v1a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V8Z"/>
        <path d="M7 14c0 1.7 1.3 3 3 3s3-1.3 3-3" strokeLinecap="round"/>
      </svg>
    ),
  },
];

export default function QuickActions() {
  // These will be passed to ChatPanel via a global event or context
  const handleAction = (label) => {
    // Dispatch to chat input
    window.dispatchEvent(new CustomEvent('quick-action', { detail: label }));
  };

  return (
    <div className="quick-actions">
      <span className="quick-actions-title">快捷问题</span>
      <div className="quick-actions-list">
        {ACTIONS.map((action) => (
          <button
            key={action.label}
            className="quick-action-btn"
            onClick={() => handleAction(action.label)}
          >
            {action.icon}
            <span>{action.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
