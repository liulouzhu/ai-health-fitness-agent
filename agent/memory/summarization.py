"""对话摘要缓冲模块"""

import os
import re
from pathlib import Path
from datetime import datetime

from agent.memory.base import MemoryAgentBase, LONGTERM_MEMORY_PATH, _memory_lock


class SummarizationManager(MemoryAgentBase):
    """对话摘要缓冲管理（基于 LangGraph state）"""

    def add_conversation_turn(self, state: dict, user_message: str, agent_response: str) -> None:
        """向 state['summary_buffer'] 添加一轮对话，并更新 turn_count"""
        if "summary_buffer" not in state:
            state["summary_buffer"] = []
        if "turn_count" not in state:
            state["turn_count"] = 0

        state["summary_buffer"].append({
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "agent": agent_response
        })
        state["turn_count"] += 1

    def should_summarize(self, state: dict, threshold: int = 10) -> bool:
        """判断是否应该进行摘要（基于 state 内的 turn_count）"""
        turn_count = state.get("turn_count", 0)
        last_summary_turn = state.get("last_summary_turn", 0)
        return (turn_count - last_summary_turn) >= threshold

    def summarize_conversations(self, state: dict, force: bool = False, max_turns: int = 20) -> dict | None:
        """对 state['summary_buffer'] 中的对话进行增量摘要，写入长期记忆文件"""
        threshold = 10
        if not force and not self.should_summarize(state, threshold):
            return None

        buffer = state.get("summary_buffer", [])
        last_summary_turn = state.get("last_summary_turn", 0)

        # 计算未摘要的部分（从 last_summary_turn 位置开始）
        # 注意：turn_count 是总轮次，buffer 长度可能小于差值（因为 buffer 可能被截断过）
        unsummarized_count = len(buffer)  # buffer 长度就是未摘要轮次（每次 add_conversation_turn 追加一条）
        if unsummarized_count == 0:
            return None

        # 最多摘要 max_turns 条（从 buffer 头部取 oldest）
        turns_to_summarize = buffer[:max_turns] if len(buffer) > max_turns else buffer

        conversation_text = "\n".join([
            f"用户: {turn['user']}\n助手: {turn['agent']}"
            for turn in turns_to_summarize
        ])

        prompt = """请对以下对话内容进行摘要，提取关键信息。

对话历史：
{conversation_history}

请以以下JSON格式返回摘要：
{{
    "summary": "用2-3句话概括本次对话的主要内容和结果",
    "learned_preferences": ["从对话中学到的用户偏好（如有）"],
    "important_facts": ["重要的用户事实或决定（如有）"],
    "topics_discussed": ["讨论的主题列表"]
}}""".format(
            conversation_history=conversation_text
        )
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        data = self._extract_json_from_response(response.content)
        if not data:
            return None

        # 写入长期记忆
        summarized_turn_count = len(turns_to_summarize)
        self._append_to_longterm_memory(data, last_summary_turn + summarized_turn_count)

        # 更新 state：移除已摘要的部分，更新 last_summary_turn
        state["summary_buffer"] = buffer[summarized_turn_count:]
        state["last_summary_turn"] = last_summary_turn + summarized_turn_count

        return {
            "summary": data.get("summary", ""),
            "learned": data.get("learned_preferences", []),
            "facts": data.get("important_facts", [])
        }

    def _get_last_summary_info(self) -> dict | None:
        """获取上次摘要的信息（从 longterm_memory.md 读取）"""
        if not os.path.exists(LONGTERM_MEMORY_PATH):
            return None

        with open(LONGTERM_MEMORY_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # 查找最新的摘要标记
        match = re.search(r'<!-- last_summary_count:(\d+) -->', content)
        if match:
            return {"message_count": int(match.group(1))}
        return None

    def _append_to_longterm_memory(self, summary_data: dict, message_count: int) -> None:
        """追加摘要到长期记忆文件（线程安全 + 原子写入）"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        new_content = f"""
## 摘要 {timestamp}
<!-- last_summary_count:{message_count} -->

**概要**: {summary_data.get('summary', '')}

**学到的偏好**:
{self._format_list(summary_data.get('learned_preferences', []))}

**重要事实**:
{self._format_list(summary_data.get('important_facts', []))}

**讨论主题**: {', '.join(summary_data.get('topics_discussed', []))}
"""

        with _memory_lock:
            # 如果文件存在，读取并在开头插入（_atomic_write 内部已确保目录存在）
            if os.path.exists(LONGTERM_MEMORY_PATH):
                with open(LONGTERM_MEMORY_PATH, "r", encoding="utf-8") as f:
                    existing = f.read()
                # 保留顶部的元信息，只更新内容部分
                new_file_content = f"# 长期记忆\n<!-- last_summary_count:{message_count} -->\n" + new_content + "\n---\n\n" + existing
            else:
                new_file_content = f"# 长期记忆\n<!-- last_summary_count:{message_count} -->\n" + new_content

            self._atomic_write(LONGTERM_MEMORY_PATH, new_file_content)

    def _format_list(self, items: list) -> str:
        """格式化列表为markdown"""
        if not items:
            return "（无）"
        return "\n".join(f"- {item}" for item in items)

    def get_longterm_memory_context(self, limit: int = 3) -> str:
        """获取最近N条长期记忆用于上下文"""
        if not os.path.exists(LONGTERM_MEMORY_PATH):
            return ""

        with open(LONGTERM_MEMORY_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # 解析并提取最近 limit 条摘要
        # 使用 re.DOTALL 让 . 可以匹配换行符
        summaries = re.findall(r'## 摘要 (.+?)\n+.*?\*\*概要\*\*: (.+?)\n', content, re.DOTALL)

        if not summaries:
            return ""

        # 取最近的（新摘要插在文件顶部，summaries[0] 是最新的，所以取头部）
        recent = summaries[:limit]
        context_parts = []

        for date, summary in recent:
            # 清理摘要中的换行和多余空白
            summary_clean = re.sub(r'\s+', ' ', summary).strip()
            context_parts.append(f"[{date}] {summary_clean}")

        return "；".join(context_parts)
