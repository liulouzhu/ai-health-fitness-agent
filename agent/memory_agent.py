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


# 便捷函数
def get_memory_manager() -> MemoryManager:
    return MemoryManager()
