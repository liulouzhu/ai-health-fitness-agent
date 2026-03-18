from langchain_core.messages import HumanMessage
from agent.llm import get_llm
from agent.state import AgentState

FOOD_AGENT_PROMPT = """你是一个食物营养分析专家。请分析用户询问的食物并提供营养信息。

分析内容：
- 食物名称
- 热量 (kcal)
- 蛋白质 (g)
- 脂肪 (g)
- 碳水化合物 (g)

直接回复分析结果，不需要额外解释。"""


class FoodAgent:
    def __init__(self):
        self.llm = get_llm()

    def run(self, state: AgentState) -> AgentState:
        """执行食物分析"""
        image_info = state.get("image_info", {})

        if image_info.get("has_image", False):
            # 带图片的输入
            image_url = image_info.get("image_url", "")
            messages = [
                {"role": "system", "content": FOOD_AGENT_PROMPT},
                HumanMessage(
                    content=[
                        {"type": "text", "text": "请分析这张图片中的食物营养成分。"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                )
            ]
        else:
            # 纯文字输入
            messages = [
                {"role": "system", "content": FOOD_AGENT_PROMPT},
                {"role": "user", "content": f"请分析以下食物的营养成分：{state['input_message']}"}
            ]

        response = self.llm.invoke(messages)
        state["food_result"] = response.content
        state["response"] = response.content
        return state
