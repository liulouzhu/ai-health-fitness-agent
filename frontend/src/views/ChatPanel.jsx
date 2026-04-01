import { useState, useRef, useEffect, useCallback } from 'react';
import { useApp } from '../store/AppContext';
import { createChatStream, parseSSEData, uploadImage } from '../services/api';
import ChatMessage from '../components/ChatMessage';
import './ChatPanel.css';

export default function ChatPanel() {
  const { state, dispatch, loadStats } = useApp();
  const [inputValue, setInputValue] = useState('');
  const [imageUrl, setImageUrl] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const streamControllerRef = useRef(null);
  const imageUrlRef = useRef(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [state.messages, scrollToBottom]);

  // Keep imageUrlRef in sync with imageUrl state
  useEffect(() => {
    imageUrlRef.current = imageUrl;
  }, [imageUrl]);

  // Handle quick action events from sidebar
  useEffect(() => {
    const handler = (e) => {
      setInputValue(e.detail);
      inputRef.current?.focus();
    };
    window.addEventListener('quick-action', handler);
    return () => window.removeEventListener('quick-action', handler);
  }, []);

  // 处理图片上传
  const handleImageUpload = useCallback(async (file) => {
    if (!file) return;

    // 验证文件类型
    const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      setUploadError('不支持的图片格式，请上传 JPG、PNG 或 WebP 格式');
      return;
    }

    // 验证文件大小（限制 10MB）
    if (file.size > 10 * 1024 * 1024) {
      setUploadError('图片大小不能超过 10MB');
      return;
    }

    setIsUploading(true);
    setUploadError(null);

    try {
      const url = await uploadImage(file);
      setImageUrl(url);
    } catch (err) {
      setUploadError(err.message || '图片上传失败');
    } finally {
      setIsUploading(false);
    }
  }, []);

  // 拖拽事件处理
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      handleImageUpload(files[0]);
    }
  }, [handleImageUpload]);

  // 点击上传区域
  const handleUploadClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  // 文件选择
  const handleFileChange = useCallback((e) => {
    const file = e.target.files?.[0];
    if (file) {
      handleImageUpload(file);
    }
    // 重置 input 以便再次选择相同文件
    e.target.value = '';
  }, [handleImageUpload]);

  // 移除已上传的图片
  const handleRemoveImage = useCallback(() => {
    setImageUrl(null);
    setUploadError(null);
  }, []);

  const sendMessage = useCallback(async () => {
    const message = inputValue.trim();
    if (!message || state.isStreaming) return;

    // Capture imageUrl at call time via ref (avoids stale closure)
    const currentImageUrl = imageUrlRef.current;

    // Add user message with image
    dispatch({ type: 'CHAT_ADD_USER_MESSAGE', payload: { content: message, imageUrl: currentImageUrl } });

    setInputValue('');
    inputRef.current?.focus();

    try {
      const stream = createChatStream(message, imageUrlRef.current);
      streamControllerRef.current = stream;

      // Start streaming state first (no message created yet)
      dispatch({ type: 'CHAT_SET_STREAMING', payload: { isStreaming: true, intent: null } });

      // Clear image after sending
      setImageUrl(null);

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
  }, [inputValue, state.isStreaming, dispatch, loadStats]);

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
            imageUrl={msg.imageUrl}
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
        {/* 图片上传/预览区域 */}
        {isUploading && (
          <div className="chat-image-upload-row">
            <div className="chat-image-uploading">
              <span className="chat-upload-spinner" />
              <span>图片上传中...</span>
            </div>
          </div>
        )}

        {uploadError && (
          <div className="chat-image-upload-row">
            <div className="chat-upload-error">
              <span>{uploadError}</span>
              <button onClick={() => setUploadError(null)}>
                <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M15 5L5 15M5 5l10 10" strokeLinecap="round"/>
                </svg>
              </button>
            </div>
          </div>
        )}

        {imageUrl && !isUploading && (
          <div className="chat-image-preview-row">
            <div className="chat-image-preview">
              <img src={imageUrl} alt="已上传图片" />
              <button className="chat-image-remove" onClick={handleRemoveImage} title="移除图片">
                <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M15 5L5 15M5 5l10 10" strokeLinecap="round"/>
                </svg>
              </button>
            </div>
          </div>
        )}

        {/* 拖拽上传区域（未上传时显示） */}
        {!imageUrl && !isUploading && (
          <div
            className={`chat-drop-zone ${isDragging ? 'dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={handleUploadClick}
          >
            <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
              <rect x="2" y="4" width="16" height="12" rx="2"/>
              <circle cx="7" cy="8.5" r="1.5"/>
              <path d="M2 14l4-4 3 3 3-3 4 4" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <span>{isDragging ? '松开以上传图片' : '拖拽图片或点击上传'}</span>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />

        <div className="chat-input-row">
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
