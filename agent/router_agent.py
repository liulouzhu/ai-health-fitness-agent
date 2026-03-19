from agent.llm import get_llm
from agent.state import AgentState
from agent.memory_agent import get_memory_agent

INTENT_TYPES = ["food", "workout", "profile_update", "general"]

SYSTEM_PROMPT = """你是一个智能路由器，负责判断用户意图并路由到对应的Agent。

可选意图：
- "food": 食物、餐饮、卡路里、营养成分、食谱推荐、图片中的食物识别
- "workout": 锻炼、健身计划、运动建议、训练方法、热量消耗统计
- "profile_update": 用户主动更新档案信息（如"我体重变成XX了"、"我长高了"等）
- "general": 其他一般性对话、问候、无法分类的问题

判断规则：
1. 如果用户发送了图片（无论文字说什么），优先判断为 "food"（食物识别）
2. 如果用户提到体重、身高、年龄、性别、目标变化，判断为 "profile_update"
3. 如果用户问"吃什么"、"推荐食物"、"营养成分"，判断为 "food"
4. 如果用户问"怎么练"、"动作要领"、"今天运动消耗"，判断为 "workout"

只返回意图标签，不要任何解释。"""


class RouterAgent:
    def __init__(self):
        self.llm = get_llm()
        self.memory_manager = get_memory_agent()

    def check_profile(self, state: AgentState) -> AgentState:
        """检查用户档案是否完整"""
        if self.memory_manager.is_profile_complete():
            state["profile_complete"] = True
        else:
            state["profile_complete"] = False
            state["response"] = self.memory_manager.get_initial_questions()
        return state

    def classify_intent(self, state: AgentState) -> AgentState:
        """意图分类node"""
        # 如果档案不完整，先处理档案创建/更新
        if not state.get("profile_complete", True):
            # 检查是否是用户回答档案信息
            user_input = state.get("input_message", "")
            # 简单判断：如果包含数字，认为是回答档案问题
            if any(c.isdigit() for c in user_input):
                state["intent"] = "profile_update"
            else:
                state["intent"] = "general"
            return state

        image_info = state.get("image_info", {})
        if image_info.get("has_image", False):
            state["intent"] = "food"
            return state

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state["input_message"]}
        ]

        response = self.llm.invoke(messages)
        intent = response.content.strip().lower()

        if intent not in INTENT_TYPES:
            intent = "general"

        state["intent"] = intent
        return state

    def routing_func(self, state: AgentState) -> str:
        """条件路由函数"""
        return state.get("intent", "general")

    def handle_profile_update(self, state: AgentState) -> AgentState:
        """处理档案更新"""
        user_input = state.get("input_message", "")

        # 检查是创建还是更新
        if not self.memory_manager.load_profile().get("height"):
            # 创建档案
            try:
                profile = self.memory_manager.create_profile(user_input)
                state["user_profile"] = profile
                state["response"] = f"档案创建成功！\n\n{self._format_profile(profile)}"
            except Exception as e:
                state["response"] = f"抱歉，无法解析你的回答，请重新描述：{e}"
        else:
            # 更新档案
            result = self.memory_manager.update_profile(user_input)
            if result.get("changed"):
                profile = self.memory_manager.load_profile()
                state["user_profile"] = profile
                state["response"] = f"档案已更新！\n\n{self._format_profile(profile)}"
            else:
                state["response"] = result.get("message", "未检测到档案变化")

        return state

    def handle_general(self, state: AgentState) -> AgentState:
        """一般对话node"""
        if not state.get("profile_complete", True):
            state["response"] = self.memory_manager.get_initial_questions()
        else:
            state["response"] = "你好！我是你的健身健康助手。请告诉我你想查询食物营养还是健身指导？"
        return state

    def _format_profile(self, profile: dict) -> str:
        """格式化档案显示"""
        goal_names = {"cut": "减脂", "bulk": "增肌", "maintain": "维持"}
        goal = profile.get("goal", "unknown")
        goal_display = goal_names.get(goal, goal)

        return f"""- 身高：{profile.get('height')} cm
- 体重：{profile.get('weight')} kg
- 年龄：{profile.get('age')} 岁
- 性别：{profile.get('gender')}
- 目标：{goal_display}
- 每日目标热量：{profile.get('target_calories')} kcal
- 每日目标蛋白质：{profile.get('target_protein')} g"""
