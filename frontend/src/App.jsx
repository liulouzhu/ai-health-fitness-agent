import { useEffect } from 'react';
import { useApp } from './store/AppContext';
import Header from './components/Header';
import ChatPanel from './views/ChatPanel';
import Sidebar from './components/Sidebar';
import HistoryView from './views/HistoryView';
import BackendErrorBanner from './components/BackendErrorBanner';
import './styles/global.css';
import './App.css';

export default function App() {
  const { state, checkBackend } = useApp();

  useEffect(() => {
    checkBackend();
    const interval = setInterval(checkBackend, 30000);
    return () => clearInterval(interval);
  }, [checkBackend]);

  return (
    <div className="app">
      <Header />
      {state.checkingBackend ? null : !state.backendAlive ? (
        <BackendErrorBanner />
      ) : null}
      <div className="app-body">
        <main className="app-main">
          {state.activeView === 'chat' ? <ChatPanel /> : <HistoryView />}
        </main>
        <aside className="app-sidebar">
          <Sidebar />
        </aside>
      </div>
    </div>
  );
}
