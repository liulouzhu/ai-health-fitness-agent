import { createContext, useContext, useReducer, useCallback } from 'react';

const AppContext = createContext(null);

const initialState = {
  // Backend status
  backendAlive: false,
  checkingBackend: true,

  // Profile
  profile: null,
  profileLoading: false,
  profileError: null,

  // Daily stats
  stats: null,
  statsLoading: false,
  statsError: null,

  // History
  history: [],
  historyLoading: false,
  historyError: null,

  // Chat
  messages: [],
  isStreaming: false,
  streamingError: null,
  currentIntent: null,

  // UI
  activeView: 'chat', // 'chat' | 'history'
  sidebarOpen: true,
  profileModalOpen: false,
};

function reducer(state, action) {
  switch (action.type) {
    case 'BACKEND_CHECK_START':
      return { ...state, checkingBackend: true };
    case 'BACKEND_CHECK_OK':
      return { ...state, checkingBackend: false, backendAlive: true };
    case 'BACKEND_CHECK_FAIL':
      return { ...state, checkingBackend: false, backendAlive: false };

    case 'PROFILE_LOAD_START':
      return { ...state, profileLoading: true, profileError: null };
    case 'PROFILE_LOAD_OK':
      return { ...state, profileLoading: false, profile: action.payload };
    case 'PROFILE_LOAD_ERROR':
      return { ...state, profileLoading: false, profileError: action.payload };

    case 'STATS_LOAD_START':
      return { ...state, statsLoading: true, statsError: null };
    case 'STATS_LOAD_OK':
      return { ...state, statsLoading: false, stats: action.payload };
    case 'STATS_LOAD_ERROR':
      return { ...state, statsLoading: false, statsError: action.payload };

    case 'HISTORY_LOAD_START':
      return { ...state, historyLoading: true, historyError: null };
    case 'HISTORY_LOAD_OK':
      return { ...state, historyLoading: false, history: action.payload };
    case 'HISTORY_LOAD_ERROR':
      return { ...state, historyLoading: false, historyError: action.payload };

    case 'CHAT_ADD_USER_MESSAGE':
      return {
        ...state,
        messages: [
          ...state.messages,
          {
            id: Date.now(),
            role: 'user',
            content: action.payload.content,
            imageUrl: action.payload.imageUrl || null,
            intent: null,
            traces: [],
          },
        ],
        currentIntent: null,
        streamingError: null,
      };
    case 'CHAT_ADD_ASSISTANT_MESSAGE':
      return {
        ...state,
        messages: [
          ...state.messages,
          { id: Date.now(), role: 'assistant', content: '', intent: state.currentIntent, traces: [] },
        ],
        currentIntent: null,
      };
    case 'CHAT_SET_LAST_ASSISTANT_INTENT':
      return {
        ...state,
        currentIntent: action.payload,
        messages: state.messages.map((m, i) =>
          i === state.messages.length - 1 && m.role === 'assistant'
            ? { ...m, intent: action.payload }
            : m
        ),
      };
    case 'CHAT_SET_STREAMING':
      return {
        ...state,
        isStreaming: action.payload.isStreaming,
        streamingError: null,
        currentIntent: action.payload.intent ?? state.currentIntent,
      };
    case 'CHAT_STREAM_TOKEN':
      return {
        ...state,
        messages: state.messages.map((m, i) =>
          i === state.messages.length - 1 && m.role === 'assistant'
            ? { ...m, content: m.content + action.payload }
            : m
        ),
      };
    case 'CHAT_STREAM_DONE':
      return { ...state, isStreaming: false };
    case 'CHAT_STREAM_ERROR':
      return { ...state, isStreaming: false, streamingError: action.payload };
    case 'CHAT_CLEAR':
      return { ...state, messages: [], currentIntent: null };
    case 'CHAT_ADD_TRACE':
      return {
        ...state,
        messages: state.messages.map((m, i) =>
          i === state.messages.length - 1 && m.role === 'assistant'
            ? { ...m, traces: [...(m.traces || []), action.payload] }
            : m
        ),
      };

    case 'SET_VIEW':
      return { ...state, activeView: action.payload };

    case 'SET_BACKEND_STATUS':
      return { ...state, backendAlive: action.payload };

    case 'SET_PROFILE_MODAL_OPEN':
      return { ...state, profileModalOpen: action.payload };

    default:
      return state;
  }
}

export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  const checkBackend = useCallback(async () => {
    dispatch({ type: 'BACKEND_CHECK_START' });
    try {
      const { checkHealth } = await import('../services/api.js');
      await checkHealth();
      dispatch({ type: 'BACKEND_CHECK_OK' });
    } catch {
      dispatch({ type: 'BACKEND_CHECK_FAIL' });
    }
  }, []);

  const loadProfile = useCallback(async () => {
    dispatch({ type: 'PROFILE_LOAD_START' });
    try {
      const { fetchProfile } = await import('../services/api.js');
      const profile = await fetchProfile();
      dispatch({ type: 'PROFILE_LOAD_OK', payload: profile });
    } catch (err) {
      dispatch({ type: 'PROFILE_LOAD_ERROR', payload: err.message });
    }
  }, []);

  const loadStats = useCallback(async () => {
    dispatch({ type: 'STATS_LOAD_START' });
    try {
      const { fetchDailyStats } = await import('../services/api.js');
      const stats = await fetchDailyStats();
      dispatch({ type: 'STATS_LOAD_OK', payload: stats });
    } catch (err) {
      dispatch({ type: 'STATS_LOAD_ERROR', payload: err.message });
    }
  }, []);

  const loadHistory = useCallback(async () => {
    dispatch({ type: 'HISTORY_LOAD_START' });
    try {
      const { fetchHistory } = await import('../services/api.js');
      const data = await fetchHistory();
      dispatch({ type: 'HISTORY_LOAD_OK', payload: data.history || [] });
    } catch (err) {
      dispatch({ type: 'HISTORY_LOAD_ERROR', payload: err.message });
    }
  }, []);

  return (
    <AppContext.Provider
      value={{
        state,
        dispatch,
        checkBackend,
        loadProfile,
        loadStats,
        loadHistory,
        setProfileModalOpen: (open) => dispatch({ type: 'SET_PROFILE_MODAL_OPEN', payload: open }),
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
}
