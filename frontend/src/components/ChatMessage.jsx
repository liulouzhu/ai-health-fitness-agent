import { useMemo } from 'react';
import './ChatMessage.css';

const INTENT_CONFIG = {
  food: { label: '食物', color: '#7BA05B', bg: 'rgba(123,160,91,0.08)' },
  food_report: { label: '记录饮食', color: '#7BA05B', bg: 'rgba(123,160,91,0.08)' },
  workout: { label: '运动', color: '#6B9FD4', bg: 'rgba(107,159,212,0.08)' },
  workout_report: { label: '记录运动', color: '#6B9FD4', bg: 'rgba(107,159,212,0.08)' },
  recipe: { label: '食谱', color: '#C9A86C', bg: 'rgba(201,168,108,0.08)' },
  stats_query: { label: '查询统计', color: '#9E7BBD', bg: 'rgba(158,123,189,0.08)' },
  profile_update: { label: '更新档案', color: '#7BA05B', bg: 'rgba(123,160,91,0.08)' },
  confirm: { label: '确认', color: '#C96B6B', bg: 'rgba(201,107,107,0.08)' },
  general: { label: '对话', color: '#6B6B6B', bg: 'rgba(107,107,107,0.08)' },
};

function formatContent(text) {
  if (!text) return '';
  // Basic markdown-like formatting
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
    .replace(/^---$/gm, '<hr>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>')
    .replace(/^/, '<p>')
    .replace(/$/, '</p>');
}

export default function ChatMessage({ role, content, intent, isStreaming = false }) {
  const intentConfig = intent ? INTENT_CONFIG[intent] || INTENT_CONFIG.general : null;

  const formattedContent = useMemo(() => formatContent(content), [content]);

  if (role === 'user') {
    return (
      <div className="chat-message user">
        <div className="chat-message-bubble">
          <p>{content}</p>
        </div>
        <div className="chat-message-avatar user">
          <svg viewBox="0 0 20 20" fill="currentColor">
            <circle cx="10" cy="7" r="3.5"/>
            <path d="M3 17c0-3.3 3.1-6 7-6s7 2.7 7 6"/>
          </svg>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-message assistant">
      <div className="chat-message-avatar assistant">
        <svg viewBox="0 0 20 20" fill="none">
          <circle cx="10" cy="10" r="8" fill="var(--color-accent)" opacity="0.15"/>
          <path d="M10 5 C7 5, 5 7.5, 5 10 C5 13, 7 15, 10 15 C13 15, 15 13, 15 10 C15 7.5, 13 5, 10 5Z" fill="var(--color-accent)" opacity="0.6"/>
          <path d="M7.5 9.5 L9.5 12.5 L12.5 7.5" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
      <div className="chat-message-content">
        {intentConfig && (
          <div
            className="chat-message-intent"
            style={{ color: intentConfig.color, background: intentConfig.bg }}
          >
            <span>{intentConfig.label}</span>
          </div>
        )}
        <div className="chat-message-bubble assistant">
          {isStreaming && !content ? (
            <span className="chat-streaming-cursor" />
          ) : (
            <div dangerouslySetInnerHTML={{ __html: formattedContent }} />
          )}
          {isStreaming && content && <span className="chat-streaming-cursor" />}
        </div>
      </div>
    </div>
  );
}
