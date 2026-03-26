import os
import re
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from agent.llm import get_llm
from config import AgentConfig


MEMORY_PATH = "memory/memory.md"
LONGTERM_MEMORY_PATH = "memory/longterm_memory.md"
PREFERENCES_PATH = "memory/preferences.md"
DAILY_STATS_PATH = "memory/daily_stats"
PENDING_STATS_FILE = "memory/pending_stats.json"

# ============ 偏好批量整合配置 ============
PENDING_PREFERENCES_FILE = "memory/pending_preferences.json"
CONSOLIDATION_THRESHOLD = 5  # 每 N 条消息触发一次整合

# ============ 整合用的 Prompt ============
CONSOLIDATE_PREFERENCES_PROMPT = """你是一个偏好整合专家。请从以下多条用户消息中整合出完整的用户偏好。

现有偏好（参考，保持不变）：
{current_preferences}

待处理的用户消息（按时间顺序）：
{messages}

请分析所有消息，提取并返回JSON格式的完整偏好（只返回JSON）：
{{
    "food_preferences": {{
        "liked": ["喜欢的食物（去重后）"],
        "disliked": ["不喜欢或不能吃的食物（去重后）"],
        "allergies": ["食物过敏（去重后）"]
    }},
    "workout_preferences": {{
        "liked": ["喜欢的运动类型（去重后）"],
        "disliked": ["不喜欢或不适合的运动（去重后）"],
        "available_equipment": ["可用的健身设备（去重后）"],
        "limitations": ["运动限制或伤病（去重后）"]
    }},
    "dietary_restrictions": ["饮食限制（去重后）"],
    "schedule_preferences": ["作息偏好（去重后）"],
    "other": ["其他偏好（去重后）"]
}}

注意：
1. 保留现有偏好中不冲突的内容
2. 只新增新发现的信息，不要删除已有的有效信息
3. 对类似的信息进行去重合并"""

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

# ============ 对话摘要 Prompt ============
SUMMARIZE_CONVERSATION_PROMPT = """请对以下对话内容进行摘要，提取关键信息。

对话历史：
{conversation_history}

请以以下JSON格式返回摘要：
{{
    "summary": "用2-3句话概括本次对话的主要内容和结果",
    "learned_preferences": ["从对话中学到的用户偏好（如有）"],
    "important_facts": ["重要的用户事实或决定（如有）"],
    "topics_discussed": ["讨论的主题列表"]
}}"""

EXTRACT_PREFERENCES_PROMPT = """从用户消息中提取偏好信息。

用户消息：{user_message}

请分析并返回JSON格式的偏好（只返回JSON，不要其他内容）：
{{
    "food_preferences": {{
        "liked": ["喜欢的食物（如有）"],
        "disliked": ["不喜欢或不能吃的食物（如有）"],
        "allergies": ["食物过敏（如有）"]
    }},
    "workout_preferences": {{
        "liked": ["喜欢的运动类型（如有）"],
        "disliked": ["不喜欢或不适合的运动（如有）"],
        "available_equipment": ["可用的健身设备（如有）"],
        "limitations": ["运动限制或伤病（如有）"]
    }},
    "dietary_restrictions": ["饮食限制，如素食、清真等（如有）"],
    "schedule_preferences": ["时间偏好，如早起/夜猫子（如有）"],
    "other": ["其他偏好（如有）"]
}}"""

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

        # 尝试使用 JSONDecoder 来解析嵌套的 JSON
        try:
            decoder = json.JSONDecoder()
            # 从字符串开头解析一个 JSON 对象
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

    def _extract_number(self, line: str) -> str:
        """从行中提取数字（支持整数和小数）"""
        match = re.search(r'\d+\.?\d*', line)
        return match.group() if match else "0"

    def _parse_meal_line(self, line: str) -> dict | None:
        """解析餐食记录行: "- 午餐: 500 kcal, 30g蛋白"
        可能格式:
        - "- 午餐: 500 kcal, 30g蛋白"
        - "- 午餐: 500 kcal, 30g蛋白, 20g脂肪, 50g碳水"
        """
        try:
            # 移除开头的 "- "
            line = line[2:].strip()

            # 提取名称（冒号之前的部分）
            if ": " not in line:
                return None
            name, rest = line.split(": ", 1)

            meal = {"name": name.strip()}

            # 提取热量
            cal_match = re.search(r'(\d+)\s*kcal', rest)
            if cal_match:
                meal["calories"] = int(cal_match.group(1))

            # 提取蛋白质
            pro_match = re.search(r'(\d+)\s*g.*蛋白', rest)
            if pro_match:
                meal["protein"] = float(pro_match.group(1))

            # 提取脂肪（可选）
            fat_match = re.search(r'(\d+)\s*g.*脂肪', rest)
            if fat_match:
                meal["fat"] = float(fat_match.group(1))

            # 提取碳水（可选）
            carbs_match = re.search(r'(\d+)\s*g.*碳水', rest)
            if carbs_match:
                meal["carbs"] = float(carbs_match.group(1))

            return meal if "calories" in meal or "protein" in meal else None
        except Exception:
            return None

    def _parse_workout_line(self, line: str) -> dict | None:
        """解析运动记录行: "- 跑步: 30分钟, 300 kcal"
        可能格式:
        - "- 跑步: 30分钟, 300 kcal"
        - "- 力量训练: 45分钟, 200 kcal"
        """
        try:
            # 移除开头的 "- "
            line = line[2:].strip()

            # 提取名称（冒号之前的部分）
            if ": " not in line:
                return None
            name, rest = line.split(": ", 1)

            workout = {"type": name.strip()}

            # 提取时长
            duration_match = re.search(r'(\d+)\s*分钟', rest)
            if duration_match:
                workout["duration"] = int(duration_match.group(1))

            # 提取热量
            cal_match = re.search(r'(\d+)\s*kcal', rest)
            if cal_match:
                workout["calories"] = int(cal_match.group(1))

            return workout if "duration" in workout or "calories" in workout else None
        except Exception:
            return None

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

    # ============ 用户偏好管理 ============

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
            "food_preferences": {
                "liked": [],
                "disliked": [],
                "allergies": []
            },
            "workout_preferences": {
                "liked": [],
                "disliked": [],
                "available_equipment": [],
                "limitations": []
            },
            "dietary_restrictions": [],
            "schedule_preferences": [],
            "other": []
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
                    prefs["food_preferences"]["liked"] = [s.strip() for s in items.split(",")]
            elif line.startswith("- 不喜欢:"):
                items = line.replace("- 不喜欢:", "").strip()
                if items and items != "（无）":
                    prefs["food_preferences"]["disliked"] = [s.strip() for s in items.split(",")]
            elif line.startswith("- 过敏:"):
                items = line.replace("- 过敏:", "").strip()
                if items and items != "（无）":
                    prefs["food_preferences"]["allergies"] = [s.strip() for s in items.split(",")]
            elif line.startswith("- 喜欢的运动:"):
                items = line.replace("- 喜欢的运动:", "").strip()
                if items and items != "（无）":
                    prefs["workout_preferences"]["liked"] = [s.strip() for s in items.split(",")]
            elif line.startswith("- 不喜欢的运动:"):
                items = line.replace("- 不喜欢的运动:", "").strip()
                if items and items != "（无）":
                    prefs["workout_preferences"]["disliked"] = [s.strip() for s in items.split(",")]
            elif line.startswith("- 可用设备:"):
                items = line.replace("- 可用设备:", "").strip()
                if items and items != "（无）":
                    prefs["workout_preferences"]["available_equipment"] = [s.strip() for s in items.split(",")]
            elif line.startswith("- 运动限制:"):
                items = line.replace("- 运动限制:", "").strip()
                if items and items != "（无）":
                    prefs["workout_preferences"]["limitations"] = [s.strip() for s in items.split(",")]
            elif line.startswith("- 饮食限制:"):
                items = line.replace("- 饮食限制:", "").strip()
                if items and items != "（无）":
                    prefs["dietary_restrictions"] = [s.strip() for s in items.split(",")]
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
        lines.append(f"- 喜欢: {', '.join(prefs['food_preferences']['liked']) or '（无）'}")
        lines.append(f"- 不喜欢: {', '.join(prefs['food_preferences']['disliked']) or '（无）'}")
        lines.append(f"- 过敏: {', '.join(prefs['food_preferences']['allergies']) or '（无）'}")
        lines.append("")

        lines.append("## 运动偏好")
        lines.append(f"- 喜欢的运动: {', '.join(prefs['workout_preferences']['liked']) or '（无）'}")
        lines.append(f"- 不喜欢的运动: {', '.join(prefs['workout_preferences']['disliked']) or '（无）'}")
        lines.append(f"- 可用设备: {', '.join(prefs['workout_preferences']['available_equipment']) or '（无）'}")
        lines.append(f"- 运动限制: {', '.join(prefs['workout_preferences']['limitations']) or '（无）'}")
        lines.append("")

        lines.append("## 饮食限制")
        lines.append(f"- 饮食限制: {', '.join(prefs['dietary_restrictions']) or '（无）'}")
        lines.append("")

        lines.append("## 作息偏好")
        lines.append(f"- 作息偏好: {', '.join(prefs['schedule_preferences']) or '（无）'}")
        lines.append("")

        lines.append("## 其他")
        lines.append(f"- 其他: {', '.join(prefs['other']) or '（无）'}")

        return "\n".join(lines)

    def save_preferences(self, prefs: dict) -> None:
        """保存用户偏好"""
        Path(PREFERENCES_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(PREFERENCES_PATH, "w", encoding="utf-8") as f:
            f.write(self._preferences_to_markdown(prefs))

    # ============ 偏好批量整合 ============

    def _ensure_pending_preferences_file(self) -> None:
        """确保待整合文件存在"""
        Path(PENDING_PREFERENCES_FILE).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(PENDING_PREFERENCES_FILE):
            with open(PENDING_PREFERENCES_FILE, "w", encoding="utf-8") as f:
                json.dump({"pending_messages": [], "last_consolidation_count": 0}, f)

    def add_pending_preference(self, user_message: str) -> None:
        """添加用户消息到待整合缓冲区"""
        self._ensure_pending_preferences_file()

        with open(PENDING_PREFERENCES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["pending_messages"].append({
            "timestamp": datetime.now().isoformat(),
            "content": user_message
        })

        with open(PENDING_PREFERENCES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def should_consolidate_preferences(self) -> bool:
        """检查是否应该触发偏好整合（达到阈值）"""
        self._ensure_pending_preferences_file()

        with open(PENDING_PREFERENCES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        pending_count = len(data["pending_messages"])
        return pending_count >= CONSOLIDATION_THRESHOLD

    def consolidate_preferences(self, force: bool = False) -> dict | None:
        """批量整合待处理的偏好消息（只调用一次 LLM）"""
        self._ensure_pending_preferences_file()

        with open(PENDING_PREFERENCES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        pending_messages = data["pending_messages"]
        if not pending_messages:
            return None

        # 检查是否达到阈值（除非强制整合）
        if not force and len(pending_messages) < CONSOLIDATION_THRESHOLD:
            return None

        # 构建消息文本
        messages_text = "\n".join([
            f"[{i+1}] {msg['timestamp']}: {msg['content']}"
            for i, msg in enumerate(pending_messages)
        ])

        # 获取当前偏好作为上下文
        current_prefs = self.load_preferences()
        current_prefs_text = json.dumps(current_prefs, ensure_ascii=False, indent=2)

        # 调用一次 LLM 整合所有消息
        prompt = CONSOLIDATE_PREFERENCES_PROMPT.format(
            current_preferences=current_prefs_text,
            messages=messages_text
        )

        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            consolidated = self._extract_json_from_response(response.content)

            if not consolidated:
                return {"success": False, "reason": "LLM 返回格式错误"}

            # 合并到现有偏好
            self._merge_preferences(consolidated)

            # 清空 pending buffer，更新计数
            data["pending_messages"] = []
            data["last_consolidation_count"] = self.get_message_count()

            with open(PENDING_PREFERENCES_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return {
                "success": True,
                "consolidated_count": len(pending_messages),
                "preferences": consolidated
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
            ("liked", "allergies"),   # 对某食物过敏则不可能喜欢它
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
                    if key == other_key and item in current["food_preferences"][conflict_key]:
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
                    if key == other_key and item in current["workout_preferences"][conflict_key]:
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

    def extract_and_save_preferences(self, user_message: str, auto_consolidate: bool = True) -> dict:
        """从用户消息中提取偏好并保存（批量模式）

        Args:
            user_message: 用户消息
            auto_consolidate: 是否自动触发批量整合
        """
        # 先把消息加入 pending buffer
        self.add_pending_preference(user_message)

        # 检查是否达到整合阈值
        if auto_consolidate and self.should_consolidate_preferences():
            result = self.consolidate_preferences()
            if result and result.get("success"):
                return {
                    "updated": True,
                    "consolidated": True,
                    "count": result["consolidated_count"],
                    "preferences": result["preferences"]
                }

        return {
            "updated": True,
            "consolidated": False,
            "pending_count": len(self._get_pending_messages())
        }

    def _get_pending_messages(self) -> list:
        """获取当前待整合的消息"""
        self._ensure_pending_preferences_file()
        with open(PENDING_PREFERENCES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("pending_messages", [])

    def get_preferences_for_context(self) -> str:
        """获取偏好字符串用于上下文注入"""
        prefs = self.load_preferences()
        context_parts = []

        # 食物偏好
        if prefs["food_preferences"]["disliked"]:
            context_parts.append(f"食物禁忌: {', '.join(prefs['food_preferences']['disliked'])}")
        if prefs["food_preferences"]["allergies"]:
            context_parts.append(f"食物过敏: {', '.join(prefs['food_preferences']['allergies'])}")
        if prefs["food_preferences"]["liked"]:
            context_parts.append(f"喜欢食物: {', '.join(prefs['food_preferences']['liked'])}")

        # 运动偏好
        if prefs["workout_preferences"]["disliked"]:
            context_parts.append(f"不喜欢运动: {', '.join(prefs['workout_preferences']['disliked'])}")
        if prefs["workout_preferences"]["limitations"]:
            context_parts.append(f"运动限制: {', '.join(prefs['workout_preferences']['limitations'])}")

        # 饮食限制
        if prefs["dietary_restrictions"]:
            context_parts.append(f"饮食限制: {', '.join(prefs['dietary_restrictions'])}")

        return "；".join(context_parts) if context_parts else ""

    # ============ State-based 摘要缓冲（替代 JSON history） ============
    # 注意：以下方法现在基于 LangGraph state 工作，不再读写 JSON 文件

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

        prompt = SUMMARIZE_CONVERSATION_PROMPT.format(
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
        """追加摘要到长期记忆文件"""
        Path(LONGTERM_MEMORY_PATH).parent.mkdir(parents=True, exist_ok=True)

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

        # 如果文件存在，读取并在开头插入
        if os.path.exists(LONGTERM_MEMORY_PATH):
            with open(LONGTERM_MEMORY_PATH, "r", encoding="utf-8") as f:
                existing = f.read()
            # 保留顶部的元信息，只更新内容部分
            new_file_content = f"# 长期记忆\n<!-- last_summary_count:{message_count} -->\n" + new_content + "\n---\n\n" + existing
        else:
            new_file_content = f"# 长期记忆\n<!-- last_summary_count:{message_count} -->\n" + new_content

        with open(LONGTERM_MEMORY_PATH, "w", encoding="utf-8") as f:
            f.write(new_file_content)

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


# 便捷函数
def get_memory_agent() -> MemoryManager:
    return MemoryManager()
