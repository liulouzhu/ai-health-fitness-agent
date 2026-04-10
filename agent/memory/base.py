"""Memory Agent 基类 - 包含公共方法和常量"""

import os
import re
import json
import tempfile
import threading
from pathlib import Path
from datetime import datetime

from agent.llm import get_llm

# 模块级全局锁：保护所有 memory 文件的并发读写
_memory_lock = threading.RLock()


# ============ 路径常量 ============
MEMORY_PATH = "memory/memory.md"
LONGTERM_MEMORY_PATH = "memory/longterm_memory.md"
PREFERENCES_PATH = "memory/preferences.md"
DAILY_STATS_PATH = "memory/daily_stats"
PENDING_STATS_FILE = "memory/pending_stats.json"
PENDING_PREFERENCES_FILE = "memory/pending_preferences.json"

# ============ 模板定义 ============
MEAL_ENTRY_TEMPLATE = "- {name}: {calories} kcal, {protein}g蛋白"
WORKOUT_ENTRY_TEMPLATE = "- {type}: {duration}分钟, {calories} kcal"
DAILY_STATS_TEMPLATE = """# 每日统计 {date}

## 基础数据
- 总热量: {consumed_calories} kcal
- 总蛋白质: {consumed_protein} g
- 总脂肪: {consumed_fat} g
- 总碳水: {consumed_carbs} g
- 总消耗: {burned_calories} kcal

## 餐食记录
{meals}

## 运动记录
{workouts}

## 剩余
- 剩余热量: {remaining_calories} kcal
- 剩余蛋白质: {remaining_protein} g
"""

INITIAL_QUESTIONS = """你好！我是你的健身健康助手。为了给你提供更好的服务，请告诉我以下信息：

1. 身高（cm）
2. 体重（kg）
3. 年龄
4. 性别
5. 健身目标（减脂 / 增肌 / 维持）

请直接回复，例如：身高175，体重70，年龄25，性别男，目标减脂"""


class MemoryAgentBase:
    """Memory Agent 基类，提供公共方法"""

    def __init__(self):
        self.llm = get_llm()
        self.memory_path = MEMORY_PATH
        self.longterm_memory_path = LONGTERM_MEMORY_PATH
        self.preferences_path = PREFERENCES_PATH
        self.daily_stats_path = DAILY_STATS_PATH
        self.pending_stats_file = PENDING_STATS_FILE
        self._ensure_memory_file()

    def _ensure_memory_file(self):
        """确保memory文件存在（线程安全初始化：双重检查 + 原子写入）

        注意：此方法在 __init__ 中调用时尚未持有锁，所以用双重检查
        模式（DCLP）：先快速路径检查，必要时再进入临界区。
        """
        Path(self.memory_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.memory_path):
            initial = "# 用户档案\n\n- user_id: default\n"
            # 双重检查：即使两个请求同时发现文件不存在，也只有其中一个能成功写入
            with _memory_lock:
                if not os.path.exists(self.memory_path):
                    self._atomic_write(self.memory_path, initial)

    # ============ 工具方法 ============

    def _atomic_write(self, file_path: str | Path, content: str) -> None:
        """原子写入：临时文件 + os.replace 原子替换

        保证写入过程中崩溃不会留下半写文件，也不会在并发时
        出现一个请求覆盖另一个请求的内容。
        """
        file_path = str(file_path)
        dir_path = os.path.dirname(file_path) or "."
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, file_path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def _parse_markdown(self, content: str) -> dict:
        """解析markdown格式的档案"""
        profile = {}
        lines = content.split("\n")
        for line in lines:
            if ": " in line:
                key, value = line.split(": ", 1)
                key = key.replace("-", "").strip()
                value = value.strip()
                profile[key] = value
        return profile

    def _dict_to_markdown(self, profile: dict) -> str:
        """将字典转换为markdown格式"""
        lines = ["# 用户档案", ""]
        for key, value in profile.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _extract_json_from_response(self, response: str) -> dict:
        """从LLM回复中提取JSON"""
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 尝试使用 JSONDecoder 来解析嵌套的 JSON
        try:
            decoder = json.JSONDecoder()
            obj, idx = decoder.raw_decode(response)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

        # 如果以上都失败，尝试从 markdown code block 中提取
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return {}

    def _extract_number(self, line: str) -> str:
        """从行中提取数字（支持整数和小数）"""
        match = re.search(r'\d+\.?\d*', line)
        return match.group() if match else "0"

    def _split_dashed_line(self, line: str) -> tuple[str, str] | None:
        """解析以 "- " 开头的行，返回 (名称, 剩余内容) 或 None"""
        if not line.startswith("- "):
            return None
        stripped = line[2:].strip()
        if ": " not in stripped:
            return None
        name, rest = stripped.split(": ", 1)
        return name.strip(), rest

    def _parse_meal_line(self, line: str) -> dict | None:
        """解析餐食记录行: "- 午餐: 500 kcal, 30g蛋白"等"""
        result = self._split_dashed_line(line)
        if result is None:
            return None
        name, rest = result
        meal = {"name": name}

        cal_match = re.search(r'(\d+)\s*kcal', rest)
        if cal_match:
            meal["calories"] = int(cal_match.group(1))
        pro_match = re.search(r'(\d+)\s*g.*蛋白', rest)
        if pro_match:
            meal["protein"] = float(pro_match.group(1))
        fat_match = re.search(r'(\d+)\s*g.*脂肪', rest)
        if fat_match:
            meal["fat"] = float(fat_match.group(1))
        carbs_match = re.search(r'(\d+)\s*g.*碳水', rest)
        if carbs_match:
            meal["carbs"] = float(carbs_match.group(1))

        return meal if "calories" in meal or "protein" in meal else None

    def _parse_workout_line(self, line: str) -> dict | None:
        """解析运动记录行: "- 跑步: 30分钟, 300 kcal"等"""
        result = self._split_dashed_line(line)
        if result is None:
            return None
        name, rest = result
        workout = {"type": name}

        duration_match = re.search(r'(\d+)\s*分钟', rest)
        if duration_match:
            workout["duration"] = int(duration_match.group(1))
        cal_match = re.search(r'(\d+)\s*kcal', rest)
        if cal_match:
            workout["calories"] = int(cal_match.group(1))

        return workout if "duration" in workout or "calories" in workout else None
