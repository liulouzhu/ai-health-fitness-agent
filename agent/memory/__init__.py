"""Memory module - 用户记忆管理子模块

包含：
- base.py: 基类和公共方法
- profile_manager.py: 用户档案管理
- daily_stats.py: 每日统计管理
- preferences.py: 偏好管理
- summarization.py: 对话摘要缓冲
"""

from agent.memory.base import MemoryAgentBase, MEMORY_PATH, PREFERENCES_PATH
from agent.memory.profile_manager import ProfileManager
from agent.memory.daily_stats import DailyStatsManager
from agent.memory.preferences import PreferencesManager, classify_preference_signal, PENDING_PREFERENCES_FILE, SIGNAL_TYPES
from agent.memory.summarization import SummarizationManager


class MemoryAgent(
    ProfileManager,
    DailyStatsManager,
    PreferencesManager,
    SummarizationManager
):
    """组合所有记忆管理功能"""
    pass


# 向后兼容别名
MemoryManager = MemoryAgent


_memory_agent_instance = None


def get_memory_agent() -> MemoryAgent:
    """获取 MemoryAgent 单例"""
    global _memory_agent_instance
    if _memory_agent_instance is None:
        _memory_agent_instance = MemoryAgent()
    return _memory_agent_instance
