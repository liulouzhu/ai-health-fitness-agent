import { useState, useRef, useEffect, useCallback } from 'react';
import { useApp } from '../store/AppContext';
import { createChatStream, parseSSEData } from '../services/api';
import ChatMessage from '../components/ChatMessage';
import './ChatPanel.css';

export default function ChatPanel() {
  const { state, dispatch, loadStats } = useApp();
  const [inputValue, setInputValue] = useState('');
  const [imageUrlInput, setImageUrlInput] = useState('');
  const [showImageInput, setShowImageInput] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const streamControllerRef = useRef(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [state.messages, scrollToBottom]);

  // Handle quick action events from sidebar
  useEffect(() => {
    const handler = (e) => {
      setInputValue(e.detail);
      inputRef.current?.focus();
    };
    window.addEventListener('quick-action', handler);
    return () => window.removeEventListener('quick-action', handler);
  }, []);

  const sendMessage = useCallback(async () => {
    const message = inputValue.trim();
    if (!message || state.isStreaming) return;

    const imageUrl = imageUrlInput.trim() || null;

    // Add user message
    dispatch({ type: 'CHAT_ADD_USER_MESSAGE', payload: message });

    setInputValue('');
    if (!showImageInput) setImageUrlInput('');
    inputRef.current?.focus();

    try {
      const stream = createChatStream(message, imageUrl);
      streamControllerRef.current = stream;

      // Start streaming state first (no message created yet)
      dispatch({ type: 'CHAT_SET_STREAMING', payload: { isStreaming: true, intent: null } });

      for await (const data of stream.start()) {
        const parsed = parseSSEData(data);

        if (parsed.type === 'done') {
          break;
        } else if (parsed.type === 'intent') {
          // First SSE chunk: intent arrived — create assistant message with this intent
          dispatch({
            type: 'CHAT_SET_STREAMING',
            payload: { isStreaming: true, intent: parsed.intent },
          });
          dispatch({ type: 'CHAT_ADD_ASSISTANT_MESSAGE' });
        } else if (parsed.type === 'token') {
          dispatch({
            type: 'CHAT_STREAM_TOKEN',
            payload: parsed.value,
          });
        }
      }

      dispatch({ type: 'CHAT_STREAM_DONE' });

      // Refresh stats after a successful interaction
      setTimeout(() => loadStats(), 500);
    } catch (err) {
      dispatch({ type: 'CHAT_STREAM_ERROR', payload: err.message });
      // Replace last empty assistant message with error
    }
  }, [inputValue, imageUrlInput, showImageInput, state.isStreaming, dispatch, loadStats]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleClearChat = () => {
    dispatch({ type: 'CHAT_CLEAR' });
  };

  return (
    <div className="chat-panel">
      {/* Messages + input wrapper — input stays fixed at bottom */}
      <div className="chat-body">
      {/* Chat messages area */}
      <div className="chat-messages">
        {state.messages.length === 0 && (
          <div className="chat-empty">
            <div className="chat-empty-icon">
              <svg viewBox="0 0 48 48" fill="none">
                <circle cx="24" cy="24" r="20" fill="var(--color-accent)" opacity="0.1"/>
                <path d="M24 14 C19 14, 16 18, 16 22 C16 27, 19 31, 24 31 C29 31, 32 27, 32 22 C32 18, 29 14, 24 14Z" fill="var(--color-accent)" opacity="0.4"/>
                <path d="M20 20 L23 26 L28 18" stroke="var(--color-accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
            <h3>开始你的健康之旅</h3>
            <p>输入你的问题，或者从右侧选择一个快捷问题</p>
          </div>
        )}

        {state.messages.map((msg) => (
          <ChatMessage
            key={msg.id}
            role={msg.role}
            content={msg.content}
            intent={msg.intent}
          />
        ))}

        {state.isStreaming && (
          <ChatMessage
            role="assistant"
            content=""
            intent={null}
            isStreaming={true}
          />
        )}

        {state.streamingError && (
          <div className="chat-error">
            <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="10" cy="10" r="7"/>
              <path d="M10 6v5M10 13.5v.5" strokeLinecap="round"/>
            </svg>
            <span>{state.streamingError}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="chat-input-area">
        {showImageInput && (
          <div className="chat-image-input-row">
            <input
              type="text"
              className="chat-image-input"
              placeholder="输入图片 URL（可选）"
              value={imageUrlInput}
              onChange={(e) => setImageUrlInput(e.target.value)}
            />
            <button
              className="chat-image-toggle"
              onClick={() => {
                setShowImageInput(false);
                setImageUrlInput('');
              }}
            >
              <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M15 5L5 15M5 5l10 10" strokeLinecap="round"/>
              </svg>
            </button>
          </div>
        )}

        <div className="chat-input-row">
          <button
            className={`chat-image-toggle ${showImageInput ? 'active' : ''}`}
            onClick={() => setShowImageInput(!showImageInput)}
            title="添加图片"
          >
            <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
              <rect x="2" y="4" width="16" height="12" rx="2"/>
              <circle cx="7" cy="8.5" r="1.5"/>
              <path d="M2 14l4-4 3 3 3-3 4 4" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>

          <textarea
            ref={inputRef}
            className="chat-textarea"
            placeholder="输入你的问题..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={state.isStreaming}
          />

          <button
            className="chat-send-btn"
            onClick={sendMessage}
            disabled={!inputValue.trim() || state.isStreaming}
          >
            {state.isStreaming ? (
              <span className="chat-send-loading" />
            ) : (
              <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M4 10h12M12 6l4 4-4 4" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            )}
          </button>

          {state.messages.length > 0 && !state.isStreaming && (
            <button className="chat-clear-btn" onClick={handleClearChat} title="清空对话">
              <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M5 5l10 10M15 5L5 15" strokeLinecap="round"/>
              </svg>
            </button>
          )}
        </div>

        <p className="chat-input-hint">
          按 Enter 发送，Shift + Enter 换行
        </p>
      </div>
      </div>{/* end .chat-body */}
    </div>
  );
}
