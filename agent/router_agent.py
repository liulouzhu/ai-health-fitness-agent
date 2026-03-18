from agent.llm import get_llm
from agent.state import AgentState

INTENT_TYPES = ["food", "workout", "general"]

SYSTEM_PROMPT = """你是一个智能路由器，负责判断用户意图并路由到对应的Agent。

可选意图：
- "food": 食物、餐饮、卡路里、营养成分、食谱推荐、图片中的食物识别
- "workout": 锻炼、健身计划、运动建议、训练方法、热量消耗统计
- "general": 其他一般性对话、问候、无法分类的问题

判断规则：
1. 如果用户发送了图片（无论文字说什么），优先判断为 "food"（食物识别）
2. 如果用户问"吃什么"、"推荐食物"、"营养成分"，判断为 "food"
3. 如果用户问"怎么练"、"动作要领"、"今天运动消耗"，判断为 "workout"

只返回意图标签，不要任何解释。"""


class RouterAgent:
    def __init__(self):
        self.llm = get_llm()

    def classify_intent(self, state: AgentState) -> AgentState:
        """意图分类node"""
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

    def handle_general(self, state: AgentState) -> AgentState:
        """一般对话node"""
        state["response"] = "你好！我是你的健身健康助手。请告诉我你想查询食物营养还是健身指导？"
        return state
