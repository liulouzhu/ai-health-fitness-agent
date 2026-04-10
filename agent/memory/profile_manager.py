"""用户档案管理模块"""

import os
from agent.memory.base import MemoryAgentBase, INITIAL_QUESTIONS, _memory_lock


class ProfileManager(MemoryAgentBase):
    """用户档案管理"""

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

    def save_profile(self, profile: dict) -> None:
        """保存用户档案（线程安全 + 原子写入）"""
        profile["updated_at"] = self._now_str()
        content = self._dict_to_markdown(profile)
        with _memory_lock:
            self._atomic_write(self.memory_path, content)

    def create_profile(self, answer: str) -> dict:
        """从用户回答创建档案（通过 LLM 解析）"""
        prompt = """从用户回答中提取用户档案信息。

用户回答：{answer}

请以以下格式回复（只返回JSON，不要其他内容）：
{{"height": 数字, "weight": 数字, "age": 数字, "gender": "male/female", "goal": "减脂/增肌/维持"}}""".format(answer=answer)
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        data = self._extract_json_from_response(response.content)
        if not data:
            raise ValueError(f"无法解析用户回答: {answer}")

        return self._build_profile_from_data(data)

    def create_profile_structured(self, height: float, weight: float, age: int, gender: str, goal: str) -> dict:
        """从结构化数据创建档案（跳过 LLM）"""
        data = {
            "height": height,
            "weight": weight,
            "age": age,
            "gender": gender,
            "goal": goal,
        }
        return self._build_profile_from_data(data)

    def _build_profile_from_data(self, data: dict) -> dict:
        """从解析后的数据构建档案（计算目标值并保存）"""
        profile = {
            "user_id": "default",
            "height": int(data.get("height", 0)),
            "weight": float(data.get("weight", 0)),
            "age": int(data.get("age", 0)),
            "gender": data.get("gender", "unknown"),
            "goal": data.get("goal", "unknown"),
            "created_at": self._now_str(),
            "updated_at": self._now_str()
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

        prompt = """分析用户话语，提取发生变化的字段。

用户原档案：
{current_profile}

用户话语：{user_message}

请回复发生了哪些字段的变化，格式如下（只返回JSON，不要其他内容）：
{{"changed": true/false, "updates": {{"height": 数字, "weight": 数字, "age": 数字, "gender": "male/female", "goal": "减脂/增肌/维持"}}}}""".format(
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

    def _normalize_value(self, value: str, mapping: dict) -> str:
        """将中文或英文值统一为英文（通过映射表）"""
        return mapping.get(value, value)

    def _normalize_gender(self, gender: str) -> str:
        """将中文或英文性别值统一为英文"""
        return self._normalize_value(gender, {"男": "male", "男性": "male", "女": "female", "女性": "female"})

    def _normalize_goal(self, goal: str) -> str:
        """将中文或英文目标值统一为英文"""
        return self._normalize_value(goal, {"减脂": "cut", "cut": "cut", "增肌": "bulk", "bulk": "bulk", "维持": "maintain", "maintain": "maintain"})

    def _calculate_targets(self, profile: dict) -> dict:
        """计算目标热量和蛋白质（本地计算，无需 LLM）"""
        h = profile.get("height", 0)
        w = profile.get("weight", 0)
        age = profile.get("age", 0)
        gender = self._normalize_gender(profile.get("gender", "male"))
        goal = self._normalize_goal(profile.get("goal", "maintain"))

        # 基础代谢率 (BMR)
        if gender == "male":
            bmr = 66 + 13.7 * w + 5 * h - 6.8 * age
        else:
            bmr = 655 + 9.6 * w + 1.8 * h - 4.7 * age

        # 每日所需 = BMR × 1.5（轻量活动）
        tdee = bmr * 1.5

        # 根据目标调整
        goal_adjustments = {"cut": -400, "bulk": 400, "maintain": 0}
        target_calories = tdee + goal_adjustments.get(goal, 0)

        # 蛋白质：体重 × 1.6 g
        target_protein = w * 1.6

        return {
            "target_calories": int(target_calories),
            "target_protein": int(target_protein)
        }

    def is_profile_complete(self) -> bool:
        """检查档案是否完整（轻量级检查，直接读原始内容）"""
        if not os.path.exists(self.memory_path):
            return False
        with open(self.memory_path, "r", encoding="utf-8") as f:
            content = f.read()
        required = ["height", "weight", "age", "gender", "goal"]
        for key in required:
            # 格式: "- key: value"，检查 key 是否存在且值不是占位符
            pattern = f"- {key}: "
            idx = content.find(pattern)
            if idx == -1:
                return False
            # 检查冒号后是否有非空非占位符内容
            start = idx + len(pattern)
            line_end = content.find("\n", start)
            line_end = line_end if line_end != -1 else len(content)
            value = content[start:line_end].strip()
            if not value or value in ("0", "unknown", ""):
                return False
        return True

    def _now_str(self) -> str:
        """返回当前日期时间字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")
