"""对话摘要缓冲模块"""

import os
import re
import json
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from agent.memory.base import MemoryAgentBase, LONGTERM_MEMORY_PATH, _memory_lock
from config import AgentConfig


@dataclass
class SummaryItem:
    """结构化的摘要条目"""

    date: str  # 摘要日期 "2026-04-20"
    summary: str  # 概要文本
    learned_preferences: list[str] = field(default_factory=list)
    important_facts: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    raw_block: str = ""  # 原始 markdown 块（用于调试）


class SummarizationManager(MemoryAgentBase):
    """对话摘要缓冲管理（基于 LangGraph state）"""

    def add_conversation_turn(
        self, state: dict, user_message: str, agent_response: str
    ) -> None:
        """向 state['summary_buffer'] 添加一轮对话，并更新 turn_count"""
        if "summary_buffer" not in state:
            state["summary_buffer"] = []
        if "turn_count" not in state:
            state["turn_count"] = 0

        state["summary_buffer"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "user": user_message,
                "agent": agent_response,
            }
        )
        state["turn_count"] += 1

    def should_summarize(self, state: dict, threshold: int = 10) -> bool:
        """判断是否应该进行摘要（基于 state 内的 turn_count）"""
        turn_count = state.get("turn_count", 0)
        last_summary_turn = state.get("last_summary_turn", 0)
        return (turn_count - last_summary_turn) >= threshold

    def summarize_conversations(
        self, state: dict, force: bool = False, max_turns: int = 20
    ) -> dict | None:
        """对 state['summary_buffer'] 中的对话进行增量摘要，写入长期记忆文件"""
        threshold = 10
        if not force and not self.should_summarize(state, threshold):
            return None

        buffer = state.get("summary_buffer", [])
        last_summary_turn = state.get("last_summary_turn", 0)

        # 计算未摘要的部分（从 last_summary_turn 位置开始）
        # 注意：turn_count 是总轮次，buffer 长度可能小于差值（因为 buffer 可能被截断过）
        unsummarized_count = len(
            buffer
        )  # buffer 长度就是未摘要轮次（每次 add_conversation_turn 追加一条）
        if unsummarized_count == 0:
            return None

        # 最多摘要 max_turns 条（从 buffer 头部取 oldest）
        turns_to_summarize = buffer[:max_turns] if len(buffer) > max_turns else buffer

        conversation_text = "\n".join(
            [
                f"用户: {turn['user']}\n助手: {turn['agent']}"
                for turn in turns_to_summarize
            ]
        )

        prompt = """请对以下对话内容进行摘要，提取关键信息。

对话历史：
{conversation_history}

请以以下JSON格式返回摘要：
{{
    "summary": "用2-3句话概括本次对话的主要内容和结果",
    "learned_preferences": ["从对话中学到的用户偏好（如有）"],
    "important_facts": ["重要的用户事实或决定（如有）"],
    "topics_discussed": ["讨论的主题列表"]
}}""".format(conversation_history=conversation_text)
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
            "facts": data.get("important_facts", []),
        }

    def _get_last_summary_info(self) -> dict | None:
        """获取上次摘要的信息（从 longterm_memory.md 读取）"""
        if not os.path.exists(LONGTERM_MEMORY_PATH):
            return None

        with open(LONGTERM_MEMORY_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # 查找最新的摘要标记
        match = re.search(r"<!-- last_summary_count:(\d+) -->", content)
        if match:
            return {"message_count": int(match.group(1))}
        return None

    def _append_to_longterm_memory(
        self, summary_data: dict, message_count: int
    ) -> None:
        """追加摘要到长期记忆文件（线程安全 + 原子写入）"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 构建结构化元数据（用于加权选择算法快速解析）
        meta = {
            "learned_preferences": summary_data.get("learned_preferences", []),
            "important_facts": summary_data.get("important_facts", []),
            "topics": summary_data.get("topics_discussed", []),
        }
        meta_json = json.dumps(meta, ensure_ascii=False)

        new_content = f"""
## 摘要 {timestamp}
<!-- last_summary_count:{message_count} -->

**概要**: {summary_data.get("summary", "")}

**学到的偏好**:
{self._format_list(summary_data.get("learned_preferences", []))}

**重要事实**:
{self._format_list(summary_data.get("important_facts", []))}

**讨论主题**: {", ".join(summary_data.get("topics_discussed", []))}

<!-- meta: {meta_json} -->
"""

        with _memory_lock:
            # 如果文件存在，读取并在开头插入（_atomic_write 内部已确保目录存在）
            if os.path.exists(LONGTERM_MEMORY_PATH):
                with open(LONGTERM_MEMORY_PATH, "r", encoding="utf-8") as f:
                    existing = f.read()
                # 保留顶部的元信息，只更新内容部分
                new_file_content = (
                    f"# 长期记忆\n<!-- last_summary_count:{message_count} -->\n"
                    + new_content
                    + "\n---\n\n"
                    + existing
                )
            else:
                new_file_content = (
                    f"# 长期记忆\n<!-- last_summary_count:{message_count} -->\n"
                    + new_content
                )

            self._atomic_write(LONGTERM_MEMORY_PATH, new_file_content)

    def _format_list(self, items: list) -> str:
        """格式化列表为markdown"""
        if not items:
            return "（无）"
        return "\n".join(f"- {item}" for item in items)

    # ============ 摘要解析与加权选择 ============

    def _parse_all_summaries(self) -> list[SummaryItem]:
        """解析 longterm_memory.md 中的所有摘要（兼容新旧格式）"""
        memory_path = self.longterm_memory_path
        if not os.path.exists(memory_path):
            return []

        with open(memory_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 按 ## 摘要 分割块（跳过第一个非摘要头部）
        blocks = re.split(r"(?=^## 摘要 )", content, flags=re.MULTILINE)
        summaries = []

        for block in blocks:
            if not block.startswith("## 摘要 "):
                continue

            # 提取日期
            date_match = re.match(r"## 摘要 (.+?)$", block, re.MULTILINE)
            if not date_match:
                continue
            date_str = date_match.group(1).strip()

            # 提取概要
            summary_match = re.search(r"\*\*概要\*\*: (.+?)(?:\n|$)", block)
            summary_text = summary_match.group(1).strip() if summary_match else ""

            # 尝试从 <!-- meta: {...} --> 提取结构化元数据（新格式）
            learned_prefs = []
            facts = []
            topics = []
            meta_match = re.search(r"<!-- meta: ({.*?}) -->", block, re.DOTALL)
            if meta_match:
                try:
                    meta = json.loads(meta_match.group(1))
                    learned_prefs = meta.get("learned_preferences", [])
                    facts = meta.get("important_facts", [])
                    topics = meta.get("topics", [])
                except json.JSONDecodeError:
                    pass

            # 降级：如果无 meta，从 markdown 行解析（旧格式兼容）
            if not meta_match:
                # 解析 **学到的偏好** 列表
                pref_section = re.search(
                    r"\*\*学到的偏好\*\*:\s*\n((?:- .+\n?)*)", block
                )
                if pref_section:
                    learned_prefs = [
                        line.lstrip("- ").strip()
                        for line in pref_section.group(1).strip().split("\n")
                        if line.strip() and line.strip() != "- （无）"
                    ]

                # 解析 **重要事实** 列表
                fact_section = re.search(r"\*\*重要事实\*\*:\s*\n((?:- .+\n?)*)", block)
                if fact_section:
                    facts = [
                        line.lstrip("- ").strip()
                        for line in fact_section.group(1).strip().split("\n")
                        if line.strip() and line.strip() != "- （无）"
                    ]

                # 解析 **讨论主题**
                topic_match = re.search(
                    r"\*\*讨论主题\*\*: (.+?)$", block, re.MULTILINE
                )
                if topic_match:
                    topics = [
                        t.strip() for t in topic_match.group(1).split(",") if t.strip()
                    ]

            summaries.append(
                SummaryItem(
                    date=date_str,
                    summary=summary_text,
                    learned_preferences=learned_prefs,
                    important_facts=facts,
                    topics=topics,
                    raw_block=block.strip(),
                )
            )

        return summaries

    def _compute_summary_score(
        self,
        item: SummaryItem,
        current_intent: str = "general",
    ) -> float:
        """计算摘要条目的综合评分（0.0 ~ 1.0）

        评分维度：
        - recency:       越新越高，线性衰减
        - pref_density:  学到的偏好越多越重要
        - fact_density:  重要事实越多越重要
        - topic_relevance: 与当前 intent 的主题匹配度
        """
        cfg = AgentConfig
        now = datetime.now()

        # Recency (0~1)
        try:
            item_date = datetime.strptime(item.date[:10], "%Y-%m-%d")
            days_old = (now - item_date).days
            recency = max(0.0, 1.0 - days_old / cfg.LONGTERM_MEMORY_MAX_AGE_DAYS)
        except ValueError:
            # 日期解析失败，给中等分
            recency = 0.5

        # Preference density (0~1)
        pref_density = min(1.0, len(item.learned_preferences) / 3.0)

        # Fact density (0~1)
        fact_density = min(1.0, len(item.important_facts) / 2.0)

        # Topic relevance (0~1)
        topic_relevance = self._compute_topic_relevance(item.topics, current_intent)

        return (
            cfg.LONGTERM_MEMORY_RECENCY_WEIGHT * recency
            + cfg.LONGTERM_MEMORY_PREF_WEIGHT * pref_density
            + cfg.LONGTERM_MEMORY_FACT_WEIGHT * fact_density
            + cfg.LONGTERM_MEMORY_TOPIC_WEIGHT * topic_relevance
        )

    def _compute_topic_relevance(self, topics: list[str], intent: str) -> float:
        """计算主题与意图的相关度（0~1）"""
        if not topics:
            return 0.3  # 无主题信息给中等分

        relevant_keywords = AgentConfig.INTENT_TOPIC_MAP.get(intent, [])
        if not relevant_keywords:
            return 0.5  # general 等无映射的 intent 给中等分

        topics_lower = [t.lower() for t in topics]
        keywords_lower = [k.lower() for k in relevant_keywords]

        # 计算匹配比例
        match_count = sum(
            1 for t in topics_lower if any(k in t or t in k for k in keywords_lower)
        )

        return min(1.0, match_count / max(1, len(topics)))

    def get_longterm_memory_context(
        self,
        limit: int = 3,
        current_intent: str = "general",
    ) -> str:
        """获取长期记忆上下文（加权选择最优的 N 条摘要）

        Args:
            limit: 返回条数上限
            current_intent: 当前对话意图，用于 topic_relevance 计算
        """
        summaries = self._parse_all_summaries()

        if not summaries:
            return ""

        # 计算每条摘要的综合评分
        scored = [
            (item, self._compute_summary_score(item, current_intent))
            for item in summaries
        ]

        # 按评分降序，取 top-K
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored[:limit]

        # 按时间排序返回（保持时间线可读性）
        selected.sort(key=lambda x: x[0].date)

        context_parts = []
        for item, score in selected:
            summary_clean = re.sub(r"\s+", " ", item.summary).strip()
            context_parts.append(f"[{item.date}] {summary_clean}")

        return "；".join(context_parts)
