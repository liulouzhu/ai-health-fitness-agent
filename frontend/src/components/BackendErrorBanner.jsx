import './BackendErrorBanner.css';

export default function BackendErrorBanner() {
  return (
    <div className="backend-error-banner">
      <div className="backend-error-inner">
        <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
          <circle cx="10" cy="10" r="7"/>
          <path d="M10 6v5M10 13.5v.5" strokeLinecap="round"/>
        </svg>
        <div className="backend-error-text">
          <strong>后端服务未连接</strong>
          <span>请在项目根目录运行 <code>python api.py</code> 启动后端</span>
        </div>
      </div>
    </div>
  );
}
