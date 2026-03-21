import os
import re
import json
from datetime import datetime
from pathlib import Path

from agent.llm import get_llm
from config import AgentConfig


MEMORY_PATH = "memory/memory.md"
INITIAL_QUESTIONS = """你好！我是你的健身健康助手。为了给你提供更好的服务，请告诉我以下信息：

1. 身高（cm）
2. 体重（kg）
3. 年龄
4. 性别
5. 健身目标（减脂 / 增肌 / 维持）

请直接回复，例如：身高175，体重70，年龄25，性别男，目标减脂"""

PROFILE_EXTRACT_PROMPT = """从用户回答中提取用户档案信息。

用户回答：{answer}

请以以下格式回复（只返回JSON，不要其他内容）：
{{"height": 数字, "weight": 数字, "age": 数字, "gender": "male/female", "goal": "减脂/增肌/维持"}}"""

UPDATE_EXTRACT_PROMPT = """分析用户话语，提取发生变化的字段。

用户原档案：
{current_profile}

用户话语：{user_message}

请回复发生了哪些字段的变化，格式如下（只返回JSON，不要其他内容）：
{{"changed": true/false, "updates": {{"height": 数字, "weight": 数字, "age": 数字, "gender": "male/female", "goal": "减脂/增肌/维持"}}}}"""

CALCULATE_TARGETS_PROMPT = """根据用户档案计算目标热量和蛋白质。

用户档案：
{profile}

请计算并返回JSON：
{{"target_calories": 数字, "target_protein": 数字}}

计算规则：
- 基础代谢率(BMR)：
  - 男 = 66 + 13.7×体重(kg) + 5×身高(cm) - 6.8×年龄
  - 女 = 655 + 9.6×体重(kg) + 1.8×身高(cm) - 4.7×年龄
- 每日所需 = BMR × 1.5（轻量活动）
- 减脂：每日所需 - 400
- 增肌：每日所需 + 400
- 维持：每日所需
- 蛋白质：体重(kg) × 1.6 g"""


class MemoryManager:
    def __init__(self):
        self.llm = get_llm()
        self.memory_path = MEMORY_PATH
        self._ensure_memory_file()

    def _ensure_memory_file(self):
        """确保memory文件存在"""
        Path(self.memory_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.memory_path):
            with open(self.memory_path, "w", encoding="utf-8") as f:
                f.write("# 用户档案\n\n- user_id: default\n")

    def get_initial_questions(self) -> str:
        """获取初始问题"""
        return INITIAL_QUESTIONS

    def load_profile(self) -> dict:
        """加载用户档案"""
        if not os.path.exists(self.memory_path):
            return {}

        with open(self.memory_path, "r", encoding="utf-8") as f:
            content = f.read()

        return self._parse_markdown(content)

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

        # 尝试从文本中提取JSON
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, response)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        return {}

    def save_profile(self, profile: dict) -> None:
        """保存用户档案"""
        profile["updated_at"] = datetime.now().strftime("%Y-%m-%d")
        with open(self.memory_path, "w", encoding="utf-8") as f:
            f.write(self._dict_to_markdown(profile))

    def create_profile(self, answer: str) -> dict:
        """从用户回答创建档案"""
        prompt = PROFILE_EXTRACT_PROMPT.format(answer=answer)
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        data = self._extract_json_from_response(response.content)
        if not data:
            raise ValueError(f"无法解析用户回答: {answer}")

        # 计算目标值
        profile = {
            "user_id": "default",
            "height": int(data.get("height", 0)),
            "weight": float(data.get("weight", 0)),
            "age": int(data.get("age", 0)),
            "gender": data.get("gender", "unknown"),
            "goal": data.get("goal", "unknown"),
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "updated_at": datetime.now().strftime("%Y-%m-%d")
        }

        # 计算目标热量和蛋白质
        targets = self._calculate_targets(profile)
        profile.update(targets)

        self.save_profile(profile)
        return profile

    def update_profile(self, user_message: str) -> dict:
        """从用户消息更新档案"""
        current = self.load_profile()
        if not current.get("height"):
            return {"changed": False, "message": "请先创建用户档案"}

        prompt = UPDATE_EXTRACT_PROMPT.format(
            current_profile=str(current),
            user_message=user_message
        )
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        data = self._extract_json_from_response(response.content)
        if not data.get("changed"):
            return {"changed": False, "message": "未检测到档案变化"}

        # 应用更新
        updates = data.get("updates", {})
        for key, value in updates.items():
            if key in current:
                current[key] = value

        # 重新计算目标
        targets = self._calculate_targets(current)
        current.update(targets)

        self.save_profile(current)
        return {"changed": True, "updates": updates, "profile": current}

    def _calculate_targets(self, profile: dict) -> dict:
        """计算目标热量和蛋白质"""
        prompt = CALCULATE_TARGETS_PROMPT.format(profile=profile)
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        data = self._extract_json_from_response(response.content)
        return {
            "target_calories": int(data.get("target_calories", 0)),
            "target_protein": int(data.get("target_protein", 0))
        }

    def is_profile_complete(self) -> bool:
        """检查档案是否完整"""
        profile = self.load_profile()
        required = ["height", "weight", "age", "gender", "goal"]
        return all(
            profile.get(k) and profile.get(k) != "0" and profile.get(k) != "unknown"
            for k in required
        )

    # ============ 每日统计 ============

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
        stats = self._get_empty_daily_stats()

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
                # 解析餐食记录
                pass  # 简化处理
            elif current_section == "workouts" and line.startswith("-"):
                # 解析运动记录
                pass  # 简化处理

        return stats

    def _extract_number(self, line: str) -> str:
        """从行中提取数字"""
        import re
        match = re.search(r'\d+', line)
        return match.group() if match else "0"

    def save_daily_stats(self, stats: dict, profile: dict = None) -> None:
        """保存每日统计到markdown文件"""
        # 计算剩余
        if profile:
            target_cal = int(profile.get("target_calories", 2000))
            target_pro = int(profile.get("target_protein", 100))
            remaining_cal = target_cal - stats["consumed_calories"] + stats["burned_calories"]
            remaining_pro = target_pro - stats["consumed_protein"]
        else:
            remaining_cal = 0
            remaining_pro = 0

        # 格式化餐食记录
        meals_str = "\n".join(
            MEAL_ENTRY_TEMPLATE.format(**m) if isinstance(m, dict) else f"- {m}"
            for m in stats.get("meals", [])
        ) or "（暂无）"

        # 格式化运动记录
        workouts_str = "\n".join(
            WORKOUT_ENTRY_TEMPLATE.format(**w) if isinstance(w, dict) else f"- {w}"
            for w in stats.get("workouts", [])
        ) or "（暂无）"

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
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def update_daily_stats(self, entry_type: str, entry: dict) -> dict:
        """更新每日统计

        Args:
            entry_type: "meal" 或 "workout"
            entry: {"name": "午餐", "calories": 500, "protein": 30} 或
                   {"type": "跑步", "duration": 30, "calories": 300}
        """
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

        # 剩余热量 = 目标 - 已摄入（不考虑运动，运动产生的是热量缺口）
        remaining_cal = max(0, target_cal - stats["consumed_calories"])
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
        """保存待确认数据到临时文件"""
        with open(PENDING_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(pending, f, ensure_ascii=False)

    def load_pending_stats(self) -> dict:
        """加载待确认数据"""
        if not os.path.exists(PENDING_STATS_FILE):
            return None
        with open(PENDING_STATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def clear_pending_stats(self) -> None:
        """清除待确认数据"""
        if os.path.exists(PENDING_STATS_FILE):
            os.remove(PENDING_STATS_FILE)


DAILY_STATS_PATH = "memory/daily_stats"
PENDING_STATS_FILE = "memory/pending_stats.json"
CONVERSATION_HISTORY_FILE = "memory/conversation_history.json"

INITIAL_QUESTIONS = """你好！我是你的健身健康助手。为了给你提供更好的服务，请告诉我以下信息：

1. 身高（cm）
2. 体重（kg）
3. 年龄
4. 性别
5. 健身目标（减脂 / 增肌 / 维持）

请直接回复，例如：身高175，体重70，年龄25，性别男，目标减脂"""

DAILY_STATS_TEMPLATE = """# 每日统计 {date}

## 摄入
- 总热量: {consumed_calories} kcal
- 总蛋白质: {consumed_protein} g
- 总脂肪: {consumed_fat} g
- 总碳水: {consumed_carbs} g

## 餐食记录
{meals}

## 运动消耗
- 总消耗: {burned_calories} kcal

## 运动记录
{workouts}

## 剩余（基于目标）
- 剩余热量: {remaining_calories} kcal
- 剩余蛋白质: {remaining_protein} g
"""

MEAL_ENTRY_TEMPLATE = "- {name}: {calories} kcal, {protein}g蛋白"
WORKOUT_ENTRY_TEMPLATE = "- {type}: {duration}分钟, {calories} kcal"

# 便捷函数
def get_memory_agent() -> MemoryManager:
    return MemoryManager()
