import { useEffect } from 'react';
import { useApp } from '../store/AppContext';
import TodayStats from './TodayStats';
import ProfileCard from './ProfileCard';
import QuickActions from './QuickActions';
import './Sidebar.css';

export default function Sidebar() {
  const { loadStats, loadProfile, state } = useApp();

  useEffect(() => {
    if (state.backendAlive) {
      loadStats();
      loadProfile();
    }
  }, [state.backendAlive, loadStats, loadProfile]);

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
    </div>
  );
}
