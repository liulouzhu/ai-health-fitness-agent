import { useEffect } from 'react';
import { useApp } from '../store/AppContext';
import TodayStats from './TodayStats';
import ProfileCard from './ProfileCard';
import QuickActions from './QuickActions';
import ProfileModal from './ProfileModal';
import './Sidebar.css';

export default function Sidebar() {
  const { loadStats, loadProfile, state, setProfileModalOpen } = useApp();

  useEffect(() => {
    if (state.backendAlive) {
      loadStats();
      loadProfile();
    }
  }, [state.backendAlive, loadStats, loadProfile]);

  // 空档案且从未弹过弹窗时，自动弹出
  useEffect(() => {
    if (state.profile === null && !state.profileLoading) {
      const hasShown = localStorage.getItem('profileModalShown');
      if (!hasShown) {
        setProfileModalOpen(true);
      }
    }
  }, [state.profile, state.profileLoading, setProfileModalOpen]);

  return (
    <div className="sidebar">
      <div className="sidebar-section">
        <TodayStats />
      </div>
      <div className="sidebar-divider" />
      <div className="sidebar-section">
        <ProfileCard />
      </div>
      <div className="sidebar-divider" />
      <div className="sidebar-section">
        <QuickActions />
      </div>

      {/* 空档案时渲染首次弹窗 */}
      {state.profile === null && (
        <ProfileModal isInitial={true} />
      )}
    </div>
  );
}
