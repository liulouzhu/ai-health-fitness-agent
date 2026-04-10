"""每日统计管理模块"""

import os
import re
from pathlib import Path
from datetime import datetime

from agent.memory.base import MemoryAgentBase, DAILY_STATS_PATH, PENDING_STATS_FILE, MEAL_ENTRY_TEMPLATE, WORKOUT_ENTRY_TEMPLATE, DAILY_STATS_TEMPLATE, _memory_lock
from config import AgentConfig


class DailyStatsManager(MemoryAgentBase):
    """每日统计管理"""

    def _get_daily_stats_path(self, date: str = None) -> Path:
        """获取每日统计文件路径"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        Path(DAILY_STATS_PATH).mkdir(parents=True, exist_ok=True)
        return Path(DAILY_STATS_PATH) / f"{date}.md"

    def load_daily_stats(self, date: str = None) -> dict:
        """加载每日统计"""
        file_path = self._get_daily_stats_path(date)
        if not file_path.exists():
            return self._get_empty_daily_stats(date)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self._parse_daily_stats(content)

    def _get_empty_daily_stats(self, date: str = None) -> dict:
        """获取空统计数据"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        return {
            "date": date,
            "consumed_calories": 0,
            "consumed_protein": 0,
            "consumed_fat": 0,
            "consumed_carbs": 0,
            "burned_calories": 0,
            "meals": [],
            "workouts": []
        }

    def _parse_daily_stats(self, content: str) -> dict:
        """解析markdown格式的每日统计"""
        # 先从文件内容中提取日期，避免用"今天"覆盖历史日期
        date_match = re.search(r'# 每日统计\s+(\d{4}-\d{2}-\d{2})', content)
        date = date_match.group(1) if date_match else datetime.now().strftime("%Y-%m-%d")

        stats = self._get_empty_daily_stats(date)

        lines = content.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith("## 餐食记录"):
                current_section = "meals"
                continue
            elif line.startswith("## 运动记录"):
                current_section = "workouts"
                continue
            elif line.startswith("- 总热量:"):
                stats["consumed_calories"] = int(self._extract_number(line))
            elif line.startswith("- 总蛋白质:"):
                stats["consumed_protein"] = float(self._extract_number(line))
            elif line.startswith("- 总脂肪:"):
                stats["consumed_fat"] = float(self._extract_number(line))
            elif line.startswith("- 总碳水:"):
                stats["consumed_carbs"] = float(self._extract_number(line))
            elif line.startswith("- 总消耗:"):
                stats["burned_calories"] = int(self._extract_number(line))
            elif current_section == "meals" and line.startswith("-"):
                # 解析餐食记录: "- 午餐: 500 kcal, 30g蛋白"
                meal = self._parse_meal_line(line)
                if meal:
                    stats["meals"].append(meal)
            elif current_section == "workouts" and line.startswith("-"):
                # 解析运动记录: "- 跑步: 30分钟, 300 kcal"
                workout = self._parse_workout_line(line)
                if workout:
                    stats["workouts"].append(workout)

        return stats

    def save_daily_stats(self, stats: dict, profile: dict = None) -> None:
        """保存每日统计到markdown文件"""
        # 计算剩余
        if profile:
            target_cal = int(profile.get("target_calories", AgentConfig.DEFAULT_TARGET_CALORIES))
            target_pro = int(profile.get("target_protein", AgentConfig.DEFAULT_TARGET_PROTEIN))
            remaining_cal = target_cal - stats["consumed_calories"] + stats["burned_calories"]
            remaining_pro = target_pro - stats["consumed_protein"]
        else:
            remaining_cal = 0
            remaining_pro = 0

        # 格式化餐食记录
        meals_list = []
        for m in stats.get("meals", []):
            if isinstance(m, dict):
                try:
                    meals_list.append(MEAL_ENTRY_TEMPLATE.format(
                        name=m.get("name", "未知"),
                        calories=int(m.get("calories", 0)),
                        protein=float(m.get("protein", 0))
                    ))
                except (KeyError, ValueError):
                    meals_list.append(f"- {m.get('name', '未知食物')}")
            else:
                meals_list.append(f"- {m}")
        meals_str = "\n".join(meals_list) or "（暂无）"

        # 格式化运动记录
        workouts_list = []
        for w in stats.get("workouts", []):
            if isinstance(w, dict):
                try:
                    workouts_list.append(WORKOUT_ENTRY_TEMPLATE.format(
                        type=w.get("type", "未知"),
                        duration=int(w.get("duration", 0)),
                        calories=int(w.get("calories", 0))
                    ))
                except (KeyError, ValueError):
                    workouts_list.append(f"- {w.get('type', '未知运动')}")
            else:
                workouts_list.append(f"- {w}")
        workouts_str = "\n".join(workouts_list) or "（暂无）"

        content = DAILY_STATS_TEMPLATE.format(
            date=stats.get("date", datetime.now().strftime("%Y-%m-%d")),
            consumed_calories=int(stats.get("consumed_calories", 0)),
            consumed_protein=float(stats.get("consumed_protein", 0)),
            consumed_fat=float(stats.get("consumed_fat", 0)),
            consumed_carbs=float(stats.get("consumed_carbs", 0)),
            meals=meals_str,
            burned_calories=int(stats.get("burned_calories", 0)),
            workouts=workouts_str,
            remaining_calories=remaining_cal,
            remaining_protein=remaining_pro
        )

        file_path = self._get_daily_stats_path(stats.get("date"))
        with _memory_lock:
            self._atomic_write(file_path, content)

    def update_daily_stats(self, entry_type: str, entry: dict) -> dict:
        """更新每日统计（整个 load-modify-write 在同一锁内完成，避免并发覆盖）

        Args:
            entry_type: "meal" 或 "workout"
            entry: {"name": "午餐", "calories": 500, "protein": 30} 或
                   {"type": "跑步", "duration": 30, "calories": 300}
        """
        with _memory_lock:
            today = datetime.now().strftime("%Y-%m-%d")
            stats = self.load_daily_stats(today)

            if entry_type == "meal":
                stats["consumed_calories"] += int(entry.get("calories", 0))
                stats["consumed_protein"] += float(entry.get("protein", 0))
                stats["consumed_fat"] += float(entry.get("fat", 0))
                stats["consumed_carbs"] += float(entry.get("carbs", 0))
                stats["meals"].append(entry)
            elif entry_type == "workout":
                stats["burned_calories"] += int(entry.get("calories", 0))
                stats["workouts"].append(entry)

            # 加载profile用于计算剩余
            profile = self.load_profile()
            self.save_daily_stats(stats, profile)
            return stats

    def get_daily_summary(self) -> str:
        """获取今日摘要"""
        today = datetime.now().strftime("%Y-%m-%d")
        stats = self.load_daily_stats(today)
        profile = self.load_profile()

        target_cal = int(profile.get("target_calories", 2000))
        target_pro = int(profile.get("target_protein", 100))

        # 剩余热量 = 目标 - 已摄入 + 运动消耗（与 /daily_stats 接口口径一致）
        remaining_cal = max(0, target_cal - stats["consumed_calories"] + stats["burned_calories"])
        remaining_pro = max(0, target_pro - stats["consumed_protein"])

        # 热量缺口 = 已摄入 - 运动消耗（负数表示有缺口）
        calorie_deficit = stats["consumed_calories"] - stats["burned_calories"]

        return f"""今日统计（{today}）：

**摄入**
- 热量：{int(stats['consumed_calories'])} / {target_cal} kcal
- 蛋白质：{stats['consumed_protein']:.0f} / {target_pro} g

**消耗**
- 运动消耗：{stats['burned_calories']} kcal
- 热量缺口：{calorie_deficit} kcal

**剩余**
- 剩余热量额度：{remaining_cal} kcal
- 剩余蛋白质：{remaining_pro:.0f} g"""

    # ============ 待确认数据临时存储 ============

    def save_pending_stats(self, pending: dict) -> None:
        """保存待确认数据到临时文件（原子写入）"""
        import json
        self._atomic_write(PENDING_STATS_FILE, json.dumps(pending, ensure_ascii=False))

    def load_pending_stats(self) -> dict:
        """加载待确认数据"""
        if not os.path.exists(PENDING_STATS_FILE):
            return None
        import json
        with open(PENDING_STATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def clear_pending_stats(self) -> None:
        """清除待确认数据"""
        if os.path.exists(PENDING_STATS_FILE):
            os.remove(PENDING_STATS_FILE)
