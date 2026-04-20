"""用户偏好管理模块"""

import os
import re
import json
from pathlib import Path
from datetime import datetime

from agent.memory.base import (
    MemoryAgentBase,
    PREFERENCES_PATH,
    PENDING_PREFERENCES_FILE,
    _memory_lock,
)
from config import AgentConfig


# ============ 偏好信号分类器 ============

SIGNAL_TYPES = {
    "hard_preference",  # 过敏、忌口、不能吃、受伤、运动限制、可用器械等硬约束
    "soft_preference",  # 喜欢/不喜欢/偏好/更喜欢等主观表达
    "behavior_signal",  # 长期行为信号（最近总是、经常、这周一直）
    "noise",  # 查询、确认、一次性记录，不进入偏好缓冲
}

# 硬偏好关键词（必须绝对匹配）
_HARD_PREFERENCE_PATTERNS = [
    (re.compile(r"过敏"), "food_allergy"),
    (re.compile(r"不能吃|不可以吃|禁食|忌口"), "food_restriction"),
    (re.compile(r"对?\S*过敏"), "food_allergy"),
    (re.compile(r"受伤|骨折|扭伤|拉伤"), "injury"),
    (re.compile(r"医生说|医嘱|不能做"), "medical_restriction"),
    (re.compile(r"只有\S*(器械|哑铃|杠铃|跑步机|单杠|双杠)"), "equipment"),
    (re.compile(r"家里有\S*(器械|哑铃|杠铃|跑步机|单杠|双杠)"), "equipment"),
    (re.compile(r"没有\S*(器械|设备)"), "equipment"),
]

# 行为信号关键词（持续性/习惯性表达，优先级高于 soft_preference）
_BEHAVIOR_SIGNAL_PATTERNS = [
    (re.compile(r"最近\s*(总是|一直|经常)"), "recent_habit"),
    (re.compile(r"这\s*(周|月|天)\s*(总是|一直|经常)"), "recent_habit"),
    (re.compile(r"天天|每天|每天都"), "frequent_behavior"),
    (re.compile(r"最近\s*(吃|喝|跑步|训练)"), "recent_behavior"),
    (re.compile(r"这段时间\s*(吃|喝|跑步)"), "recent_behavior"),
    (re.compile(r"最近\s*在\s*(吃|喝|跑步)"), "recent_behavior"),
    (re.compile(r"这段\s*时间\s*(总是\s*)?"), "recent_habit"),
    (re.compile(r"最近.*(喝|吃)"), "recent_behavior"),
]

# 软偏好关键词
_SOFT_PREFERENCE_PATTERNS = [
    (re.compile(r"喜欢|偏爱|偏好|更喜欢|宁愿"), "like_dislike"),
    (re.compile(r"不爱|讨厌|厌恶|不喜欢|不想吃|不愿意吃"), "like_dislike"),
    (re.compile(r"平时喜欢|比较喜欢"), "soft_preference"),
    (re.compile(r"一般.*(吃|喝|做|跑步|训练|锻炼)"), "habit"),
    (re.compile(r"平时.*(吃|喝|做|跑步|训练|锻炼)"), "habit"),
    (re.compile(r"习惯.*(吃|喝|跑步|训练)"), "habit"),
    (re.compile(r"总是.*(吃|喝|跑步|做)"), "habit"),
]

# 噪音关键词（查询、确认、一次性行为记录）
_NOISE_PATTERNS = [
    (re.compile(r"^是$|^否$|^对$|^没错$|^是的$"), "confirmation"),
    (re.compile(r"^我吃了?|^我跑了?|^我做了?|^我喝?了?"), "single_behavior"),
    (
        re.compile(
            r"看看?\s*(今天|昨天|这周|本月)?\s*(剩余|还剩|多少|热量|蛋白质|卡路里)"
        ),
        "query_stats",
    ),
    (re.compile(r"分析.*(今天|昨天|这周)?.*吃"), "query_stats"),
    (re.compile(r"今天摄入|今天吃了多少|热量多少|算一下|帮我算"), "query_stats"),
    (re.compile(r"还剩多少|剩余多少"), "query_stats"),
    (re.compile(r"安排\s*(今天|明天|这周)?\s*训练"), "request_training"),
    (
        re.compile(r"推荐\s*(高蛋白|低脂|低卡|健康)?\s*(菜谱|食谱|食物|餐)"),
        "request_recipe",
    ),
    (re.compile(r"记录"), "recording"),
    (re.compile(r"打卡"), "checkin"),
    (re.compile(r"^$"), "empty"),
]

# 整合阈值
CONSOLIDATION_THRESHOLD = 5  # 每 N 条消息触发一次整合


def classify_preference_signal(message: str) -> dict:
    """轻量级偏好信号分类器（纯规则，无 LLM 调用）

    Args:
        message: 用户消息

    Returns:
        dict: {
            "should_buffer": bool,   # 是否应该进入偏好缓冲
            "signal_type": str,      # 信号类型
            "confidence": float,     # 置信度 0.0-1.0
            "reason": str,           # 分类理由
            "matched_keyword": str,   # 命中的关键词（用于调试）
        }
    """
    if not message or not message.strip():
        return {
            "should_buffer": False,
            "signal_type": "noise",
            "confidence": 0.0,
            "reason": "空消息",
            "matched_keyword": "",
        }

    msg = message.strip()

    # 1. 先检查硬偏好（最高优先级，置信度 0.9+）
    for pattern, keyword in _HARD_PREFERENCE_PATTERNS:
        if pattern.search(msg):
            return {
                "should_buffer": True,
                "signal_type": "hard_preference",
                "confidence": 0.95,
                "reason": f"硬偏好信号：{keyword}",
                "matched_keyword": keyword,
            }

    # 2. 检查行为信号（置信度 0.5-0.7，弱信号）
    # 在 soft_preference 之前检查，因为"最近总是..."等模式会被 soft_preference 误匹配
    for pattern, keyword in _BEHAVIOR_SIGNAL_PATTERNS:
        if pattern.search(msg):
            confidence = 0.7 if keyword == "recent_habit" else 0.6
            return {
                "should_buffer": True,
                "signal_type": "behavior_signal",
                "confidence": confidence,
                "reason": f"行为信号：{keyword}（持续性行为）",
                "matched_keyword": keyword,
            }

    # 3. 检查软偏好（置信度 0.7-0.85）
    for pattern, keyword in _SOFT_PREFERENCE_PATTERNS:
        if pattern.search(msg):
            # 如果命中"一般/平时/习惯 + 动作"，置信度稍低
            confidence = 0.8 if keyword in ("habit", "like_dislike") else 0.85
            return {
                "should_buffer": True,
                "signal_type": "soft_preference",
                "confidence": confidence,
                "reason": f"软偏好信号：{keyword}",
                "matched_keyword": keyword,
            }

    # 4. 检查噪音（直接排除）
    for pattern, keyword in _NOISE_PATTERNS:
        if pattern.search(msg):
            return {
                "should_buffer": False,
                "signal_type": "noise",
                "confidence": 0.95,
                "reason": f"噪音信号：{keyword}",
                "matched_keyword": keyword,
            }

    # 5. 默认：单次行为记录不进缓冲
    # 启发式：句子短且包含"吃了/跑了/喝了"但没有"总是/最近/经常"
    if len(msg) < 30 and re.search(r"吃了|喝了|跑了|做了|运动了|训练了", msg):
        return {
            "should_buffer": False,
            "signal_type": "noise",
            "confidence": 0.7,
            "reason": "单次行为记录，不作为长期偏好",
            "matched_keyword": "single_behavior_heuristic",
        }

    # 6. 其他模糊情况：不进入缓冲，但不明确归类为噪音
    return {
        "should_buffer": False,
        "signal_type": "uncertain",
        "confidence": 0.5,
        "reason": "无法明确分类，默认不进缓冲",
        "matched_keyword": "",
    }


class PreferencesManager(MemoryAgentBase):
    """用户偏好管理"""

    def load_preferences(self) -> dict:
        """加载用户偏好"""
        if not os.path.exists(PREFERENCES_PATH):
            return self._get_empty_preferences()

        with open(PREFERENCES_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        return self._parse_preferences(content)

    def _get_empty_preferences(self) -> dict:
        """获取空偏好结构"""
        return {
            "food_preferences": {"liked": [], "disliked": [], "allergies": []},
            "workout_preferences": {
                "liked": [],
                "disliked": [],
                "available_equipment": [],
                "limitations": [],
            },
            "dietary_restrictions": [],
            "schedule_preferences": [],
            "other": [],
        }

    def _parse_preferences(self, content: str) -> dict:
        """解析偏好文件"""
        prefs = self._get_empty_preferences()

        current_section = None
        lines = content.split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("## 食物偏好"):
                current_section = "food"
            elif line.startswith("## 运动偏好"):
                current_section = "workout"
            elif line.startswith("- 喜欢:"):
                items = line.replace("- 喜欢:", "").strip()
                if items and items != "（无）":
                    prefs["food_preferences"]["liked"] = [
                        s.strip() for s in items.split(",")
                    ]
            elif line.startswith("- 不喜欢:"):
                items = line.replace("- 不喜欢:", "").strip()
                if items and items != "（无）":
                    prefs["food_preferences"]["disliked"] = [
                        s.strip() for s in items.split(",")
                    ]
            elif line.startswith("- 过敏:"):
                items = line.replace("- 过敏:", "").strip()
                if items and items != "（无）":
                    prefs["food_preferences"]["allergies"] = [
                        s.strip() for s in items.split(",")
                    ]
            elif line.startswith("- 喜欢的运动:"):
                items = line.replace("- 喜欢的运动:", "").strip()
                if items and items != "（无）":
                    prefs["workout_preferences"]["liked"] = [
                        s.strip() for s in items.split(",")
                    ]
            elif line.startswith("- 不喜欢的运动:"):
                items = line.replace("- 不喜欢的运动:", "").strip()
                if items and items != "（无）":
                    prefs["workout_preferences"]["disliked"] = [
                        s.strip() for s in items.split(",")
                    ]
            elif line.startswith("- 可用设备:"):
                items = line.replace("- 可用设备:", "").strip()
                if items and items != "（无）":
                    prefs["workout_preferences"]["available_equipment"] = [
                        s.strip() for s in items.split(",")
                    ]
            elif line.startswith("- 运动限制:"):
                items = line.replace("- 运动限制:", "").strip()
                if items and items != "（无）":
                    prefs["workout_preferences"]["limitations"] = [
                        s.strip() for s in items.split(",")
                    ]
            elif line.startswith("- 饮食限制:"):
                items = line.replace("- 饮食限制:", "").strip()
                if items and items != "（无）":
                    prefs["dietary_restrictions"] = [
                        s.strip() for s in items.split(",")
                    ]
            elif line.startswith("- 作息偏好:"):
                items = line.replace("- 作息偏好:", "").strip()
                if items and items != "（无）":
                    prefs["schedule_preferences"] = [items]
            elif line.startswith("- 其他:"):
                items = line.replace("- 其他:", "").strip()
                if items and items != "（无）":
                    prefs["other"] = [items]

        return prefs

    def _preferences_to_markdown(self, prefs: dict) -> str:
        """将偏好字典转换为markdown格式"""
        lines = ["# 用户偏好", ""]

        lines.append("## 食物偏好")
        lines.append(
            f"- 喜欢: {', '.join(prefs['food_preferences']['liked']) or '（无）'}"
        )
        lines.append(
            f"- 不喜欢: {', '.join(prefs['food_preferences']['disliked']) or '（无）'}"
        )
        lines.append(
            f"- 过敏: {', '.join(prefs['food_preferences']['allergies']) or '（无）'}"
        )
        lines.append("")

        lines.append("## 运动偏好")
        lines.append(
            f"- 喜欢的运动: {', '.join(prefs['workout_preferences']['liked']) or '（无）'}"
        )
        lines.append(
            f"- 不喜欢的运动: {', '.join(prefs['workout_preferences']['disliked']) or '（无）'}"
        )
        lines.append(
            f"- 可用设备: {', '.join(prefs['workout_preferences']['available_equipment']) or '（无）'}"
        )
        lines.append(
            f"- 运动限制: {', '.join(prefs['workout_preferences']['limitations']) or '（无）'}"
        )
        lines.append("")

        lines.append("## 饮食限制")
        lines.append(
            f"- 饮食限制: {', '.join(prefs['dietary_restrictions']) or '（无）'}"
        )
        lines.append("")

        lines.append("## 作息偏好")
        lines.append(
            f"- 作息偏好: {', '.join(prefs['schedule_preferences']) or '（无）'}"
        )
        lines.append("")

        lines.append("## 其他")
        lines.append(f"- 其他: {', '.join(prefs['other']) or '（无）'}")

        return "\n".join(lines)

    def save_preferences(self, prefs: dict) -> None:
        """保存用户偏好（线程安全 + 原子写入）"""
        Path(PREFERENCES_PATH).parent.mkdir(parents=True, exist_ok=True)
        content = self._preferences_to_markdown(prefs)
        with _memory_lock:
            self._atomic_write(PREFERENCES_PATH, content)

    # ============ 偏好批量整合 ============

    def _ensure_pending_preferences_file(self) -> None:
        """确保待整合文件存在（线程安全初始化：检查存在性 + 原子写入初始内容）

        注意：即使两个请求同时发现文件不存在，_atomic_write 的幂等性也保证
        不会写出坏 JSON。_atomic_write 本身受 _memory_lock 保护。
        """
        Path(PENDING_PREFERENCES_FILE).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(PENDING_PREFERENCES_FILE):
            initial = json.dumps(
                {"pending_messages": [], "last_consolidation_count": 0}
            )
            with _memory_lock:
                # 双重检查：持有锁后再次确认，避免另一请求已创建
                if not os.path.exists(PENDING_PREFERENCES_FILE):
                    self._atomic_write(PENDING_PREFERENCES_FILE, initial)

    def add_pending_preference(
        self, user_message: str, signal_info: dict = None
    ) -> None:
        """添加用户消息到待整合缓冲区（线程安全 read-modify-write）

        Args:
            user_message: 用户消息
            signal_info: 分类器返回的信号信息，可选（向后兼容）
        """
        self._ensure_pending_preferences_file()

        entry = {
            "timestamp": datetime.now().isoformat(),
            "content": user_message,
        }
        if signal_info:
            entry["signal_type"] = signal_info.get("signal_type", "unknown")
            entry["confidence"] = signal_info.get("confidence", 0.0)
            entry["reason"] = signal_info.get("reason", "")

        with _memory_lock:
            with open(PENDING_PREFERENCES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["pending_messages"].append(entry)
            self._atomic_write(
                PENDING_PREFERENCES_FILE, json.dumps(data, ensure_ascii=False, indent=2)
            )

    def should_consolidate_preferences(self) -> bool:
        """检查是否应该触发偏好整合（达到阈值）"""
        self._ensure_pending_preferences_file()

        with open(PENDING_PREFERENCES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        pending_count = len(data["pending_messages"])
        return pending_count >= CONSOLIDATION_THRESHOLD

    def consolidate_preferences(self, force: bool = False) -> dict | None:
        """批量整合待处理的偏好消息（整个过程在同一锁内完成，避免并发写入覆盖）

        注意：由于包含 LLM 调用，持有锁的时间较长，但 pending 文件本身
        写入频率低（每 N 条消息一次），因此可接受。
        """
        with _memory_lock:
            self._ensure_pending_preferences_file()

            with open(PENDING_PREFERENCES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            pending_messages = data["pending_messages"]
            if not pending_messages:
                return None

            # 检查是否达到阈值（除非强制整合）
            if not force and len(pending_messages) < CONSOLIDATION_THRESHOLD:
                return None

            # 构建消息文本（包含信号类型，供 LLM 参考权重）
            # 格式：[信号类型@置信度] 内容
            messages_text = "\n".join(
                [
                    f"[{msg.get('signal_type', 'unknown')}@{msg.get('confidence', 0.0):.2f}] {msg['content']}"
                    for msg in pending_messages
                ]
            )

            # 获取当前偏好作为上下文
            current_prefs = self.load_preferences()
            current_prefs_text = json.dumps(current_prefs, ensure_ascii=False, indent=2)

            # 调用一次 LLM 整合所有消息
            prompt = """你是一个偏好整合专家。请从以下多条用户消息中整合出完整的用户偏好。

信号类型说明：
- hard_preference: 明确的过敏、忌口、不能吃、受伤等硬约束（最高优先级，必须采纳）
- soft_preference: 明确的喜欢/不喜欢/偏好表达（采纳）
- behavior_signal: 长期行为信号如"最近总是跑步"，作为弱参考（谨慎采纳，需多次出现才视为偏好）

现有偏好（参考，保持不变）：
{current_preferences}

待处理的用户消息（信号类型@置信度）：
{messages}

请分析所有消息，提取并返回JSON格式的完整偏好（只返回JSON）：
{{
    "food_preferences": {{
        "liked": ["喜欢的食物（去重后，仅采纳 hard_preference 和 soft_preference）"],
        "disliked": ["不喜欢或不能吃的食物（去重后，hard_preference 必采纳）"],
        "allergies": ["食物过敏（hard_preference，去重后）"]
    }},
    "workout_preferences": {{
        "liked": ["喜欢的运动类型（soft_preference，去重后）"],
        "disliked": ["不喜欢或不适合的运动（去重后）"],
        "available_equipment": ["可用的健身设备（hard_preference，去重后）"],
        "limitations": ["运动限制或伤病（hard_preference，去重后）"]
    }},
    "dietary_restrictions": ["饮食限制（hard_preference，去重后）"],
    "schedule_preferences": ["作息偏好（soft_preference，去重后）"],
    "other": ["其他偏好（去重后）"]
}}

注意：
1. 保留现有偏好中不冲突的内容
2. 只新增新发现的硬偏好或软偏好，行为信号（behavior_signal）作为辅助参考
3. 对类似的信息进行去重合并
4. 不要把单次行为记录（如"我吃了一个鸡蛋"）当作偏好，除非该消息本身是 hard_preference 或 soft_preference""".format(
                current_preferences=current_prefs_text, messages=messages_text
            )

            try:
                response = self.llm.invoke([{"role": "user", "content": prompt}])
                consolidated = self._extract_json_from_response(response.content)

                if not consolidated:
                    return {"success": False, "reason": "LLM 返回格式错误"}

                # 合并到现有偏好（_merge_preferences 内部会调用 save_preferences，
                # save_preferences 也会再次获取 _memory_lock，但因为是 RLock 所以可重入）
                self._merge_preferences(consolidated)

                # 失效检测：检查哪些偏好可能已过时
                invalidated = self._detect_invalidated_preferences(pending_messages)
                removed = self._remove_invalidated_preferences(invalidated)

                # 清空 pending buffer，记录本次整合的消息数量
                consolidated_count = len(pending_messages)
                data["pending_messages"] = []
                data["last_consolidation_count"] = consolidated_count
                self._atomic_write(
                    PENDING_PREFERENCES_FILE,
                    json.dumps(data, ensure_ascii=False, indent=2),
                )

                return {
                    "success": True,
                    "consolidated_count": consolidated_count,
                    "preferences": consolidated,
                    "invalidated_removed": removed,
                }

            except Exception as e:
                return {"success": False, "reason": str(e)}

    def _merge_preferences(self, new_prefs: dict) -> None:
        """将新的偏好合并到现有偏好（去重追加 + 冲突消解）

        当某项出现在冲突集合中（如 liked vs disliked）时，后出现的意见覆盖先前的。
        """
        current = self.load_preferences()

        # 食物偏好
        food_conflict_pairs = [
            ("liked", "disliked"),
            ("disliked", "liked"),
            ("liked", "allergies"),  # 对某食物过敏则不可能喜欢它
            ("disliked", "allergies"),
        ]
        for key in ["liked", "disliked", "allergies"]:
            new_items = new_prefs.get("food_preferences", {}).get(key, [])
            existing = set(current["food_preferences"][key])
            for item in new_items:
                if not item or item in existing:
                    continue
                # 先从冲突集合中移除（冲突消解）
                for other_key, conflict_key in food_conflict_pairs:
                    if (
                        key == other_key
                        and item in current["food_preferences"][conflict_key]
                    ):
                        current["food_preferences"][conflict_key].remove(item)
                current["food_preferences"][key].append(item)

        # 运动偏好（liked / disliked 之间也会冲突）
        workout_conflict_pairs = [("liked", "disliked"), ("disliked", "liked")]
        for key in ["liked", "disliked", "available_equipment", "limitations"]:
            new_items = new_prefs.get("workout_preferences", {}).get(key, [])
            existing = set(current["workout_preferences"][key])
            for item in new_items:
                if not item or item in existing:
                    continue
                for other_key, conflict_key in workout_conflict_pairs:
                    if (
                        key == other_key
                        and item in current["workout_preferences"][conflict_key]
                    ):
                        current["workout_preferences"][conflict_key].remove(item)
                current["workout_preferences"][key].append(item)

        # 饮食限制：后出现的覆盖先前的（多次出现的以最后一次为准）
        new_restrictions = new_prefs.get("dietary_restrictions", [])
        existing_restrictions = set(current["dietary_restrictions"])
        for item in new_restrictions:
            if item and item not in existing_restrictions:
                current["dietary_restrictions"].append(item)

        # 作息偏好：同饮食限制
        new_schedule = new_prefs.get("schedule_preferences", [])
        for item in new_schedule:
            if item and item not in current["schedule_preferences"]:
                current["schedule_preferences"].append(item)

        # 其他偏好：同饮食限制
        new_other = new_prefs.get("other", [])
        for item in new_other:
            if item and item not in current["other"]:
                current["other"].append(item)

        self.save_preferences(current)

    def extract_and_save_preferences(
        self, user_message: str, auto_consolidate: bool = True
    ) -> dict:
        """从用户消息中提取偏好并保存（批量模式，带分类 gate）

        Args:
            user_message: 用户消息
            auto_consolidate: 是否自动触发批量整合

        Returns:
            dict: {
                "updated": bool,       # 是否有更新
                "skipped": bool,       # 是否被跳过（噪音）
                "consolidated": bool,  # 是否触发了批量整合
                "pending_count": int,  # 当前 pending 数量
                "signal_type": str,    # 信号类型
                "confidence": float,   # 置信度
                "reason": str,         # 分类理由
            }
        """
        # 先做轻量分类
        signal = classify_preference_signal(user_message)

        if not signal["should_buffer"]:
            return {
                "updated": False,
                "skipped": True,
                "consolidated": False,
                "pending_count": self.get_message_count(),
                "signal_type": signal["signal_type"],
                "confidence": signal["confidence"],
                "reason": signal["reason"],
            }

        # 只有应该缓冲的才加入 pending buffer
        self.add_pending_preference(user_message, signal)

        # 检查是否达到整合阈值
        if auto_consolidate and self.should_consolidate_preferences():
            result = self.consolidate_preferences()
            if result and result.get("success"):
                return {
                    "updated": True,
                    "skipped": False,
                    "consolidated": True,
                    "count": result["consolidated_count"],
                    "preferences": result["preferences"],
                    "signal_type": signal["signal_type"],
                    "confidence": signal["confidence"],
                    "reason": signal["reason"],
                    "pending_count": self.get_message_count(),
                }

        return {
            "updated": True,
            "skipped": False,
            "consolidated": False,
            "pending_count": self.get_message_count(),
            "signal_type": signal["signal_type"],
            "confidence": signal["confidence"],
            "reason": signal["reason"],
        }

    def _get_pending_messages(self) -> list:
        """获取当前待整合的消息"""
        self._ensure_pending_preferences_file()
        with open(PENDING_PREFERENCES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("pending_messages", [])

    def get_message_count(self) -> int:
        """获取 pending_messages 的总条数"""
        return len(self._get_pending_messages())

    def get_preferences_for_context(self) -> str:
        """获取偏好字符串用于上下文注入"""
        prefs = self.load_preferences()
        context_parts = []

        # 食物偏好
        if prefs["food_preferences"]["disliked"]:
            context_parts.append(
                f"食物禁忌: {', '.join(prefs['food_preferences']['disliked'])}"
            )
        if prefs["food_preferences"]["allergies"]:
            context_parts.append(
                f"食物过敏: {', '.join(prefs['food_preferences']['allergies'])}"
            )
        if prefs["food_preferences"]["liked"]:
            context_parts.append(
                f"喜欢食物: {', '.join(prefs['food_preferences']['liked'])}"
            )

        # 运动偏好
        if prefs["workout_preferences"]["disliked"]:
            context_parts.append(
                f"不喜欢运动: {', '.join(prefs['workout_preferences']['disliked'])}"
            )
        if prefs["workout_preferences"]["limitations"]:
            context_parts.append(
                f"运动限制: {', '.join(prefs['workout_preferences']['limitations'])}"
            )

        # 饮食限制
        if prefs["dietary_restrictions"]:
            context_parts.append(
                f"饮食限制: {', '.join(prefs['dietary_restrictions'])}"
            )

        return "；".join(context_parts) if context_parts else ""

    # ============ 偏好失效检测 ============

    def _detect_invalidated_preferences(self, new_messages: list[dict]) -> list[dict]:
        """用 LLM 检测哪些偏好可能已失效

        Returns:
            list[dict]: [{"category": "...", "item": "...", "reason": "...", "confidence": 0.0-1.0}]
        """
        current_prefs = self.load_preferences()
        current_prefs_text = json.dumps(current_prefs, ensure_ascii=False, indent=2)

        messages_text = "\n".join(
            [m.get("content", "") for m in new_messages if m.get("content")]
        )
        if not messages_text.strip():
            return []

        prompt = """你是一个偏好验证专家。请分析用户最新消息，判断哪些已有偏好可能已失效。

当前偏好：
{current_prefs}

用户最新消息：
{messages}

请返回 JSON 数组，列出可能失效的偏好：
[
    {{
        "category": "food_allergy|food_restriction|workout_limitation|dietary_restriction",
        "item": "已失效的条目（精确匹配现有偏好中的文本）",
        "reason": "失效原因（引用用户原话）",
        "confidence": 0.0-1.0
    }}
]

判断规则：
- 用户明确表示不再过敏/限制（如"我脱敏了"、"现在可以吃了"） → confidence 0.9+
- 用户行为与偏好矛盾（如过敏物但最近在吃） → confidence 0.5-0.7
- 仅是提及，无明确改变信号 → 不列入

如果没有失效偏好，返回空数组 []""".format(
            current_prefs=current_prefs_text, messages=messages_text
        )

        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            result = self._extract_json_from_response(response.content)
            return result if isinstance(result, list) else []
        except Exception:
            return []

    def _remove_invalidated_preferences(self, invalidated: list[dict]) -> list[str]:
        """从偏好文件中移除已失效的条目

        Returns:
            list[str]: 被移除的条目列表
        """
        if not invalidated:
            return []

        current = self.load_preferences()
        removed = []
        threshold = AgentConfig.PREFERENCE_INVALIDATION_CONFIDENCE

        for item in invalidated:
            category = item.get("category", "")
            target = item.get("item", "").strip()
            confidence = item.get("confidence", 0)

            # 只处理高置信度的失效
            if not target or confidence < threshold:
                continue

            # 按分类移除
            if (
                category == "food_allergy"
                and target in current["food_preferences"]["allergies"]
            ):
                current["food_preferences"]["allergies"].remove(target)
                removed.append(f"过敏: {target}")
            elif (
                category == "food_restriction"
                and target in current["food_preferences"]["disliked"]
            ):
                current["food_preferences"]["disliked"].remove(target)
                removed.append(f"食物禁忌: {target}")
            elif (
                category == "workout_limitation"
                and target in current["workout_preferences"]["limitations"]
            ):
                current["workout_preferences"]["limitations"].remove(target)
                removed.append(f"运动限制: {target}")
            elif (
                category == "dietary_restriction"
                and target in current["dietary_restrictions"]
            ):
                current["dietary_restrictions"].remove(target)
                removed.append(f"饮食限制: {target}")

        if removed:
            self.save_preferences(current)

        return removed
