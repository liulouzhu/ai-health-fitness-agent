import { useApp } from '../store/AppContext';
import './Header.css';

export default function Header() {
  const { state, dispatch } = useApp();

  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-brand">
          <svg className="header-logo" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="16" cy="16" r="14" fill="var(--color-accent)" opacity="0.12"/>
            <path d="M16 8 C12 8, 9 11, 9 15 C9 19, 12 22, 16 22 C20 22, 23 19, 23 15 C23 11, 20 8, 16 8Z" fill="var(--color-accent)" opacity="0.5"/>
            <path d="M13 14 L15 18 L19 12" stroke="var(--color-accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <div className="header-title-group">
            <span className="header-title">健身健康助手</span>
            <span className="header-subtitle">AI Fitness Companion</span>
          </div>
        </div>

        <nav className="header-nav">
          <button
            className={`header-nav-btn ${state.activeView === 'chat' ? 'active' : ''}`}
            onClick={() => dispatch({ type: 'SET_VIEW', payload: 'chat' })}
          >
            <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M3 8a5 5 0 0 1 10 0v1a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V8Z" strokeLinecap="round"/>
              <path d="M8 5v6M5 8h6" strokeLinecap="round"/>
            </svg>
            对话
          </button>
          <button
            className={`header-nav-btn ${state.activeView === 'history' ? 'active' : ''}`}
            onClick={() => dispatch({ type: 'SET_VIEW', payload: 'history' })}
          >
            <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="10" cy="10" r="7"/>
              <path d="M10 6v4l3 3" strokeLinecap="round"/>
            </svg>
            历史
          </button>
        </nav>
      </div>
    </header>
  );
}
