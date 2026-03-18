from llm import get_llm

IntentType = ["food", "workout", "general"]
system_prompt = """
你是一个智能路由器，负责根据用户的输入消息判断用户的意图，并将其路由到适当的代理进行处理。你需要分析用户的消息内容，识别出用户的意图，并返回一个明确的意图标签。以下是一些可能的意图类型：
如果用户问的是与食物、餐饮、卡路里、营养成分、图片中的食物识别相关的问题，则返回 "food"，
如果用户问的是与锻炼、健身计划、运动建议、训练方法相关的问题，则返回 "workout"，
如果用户问的是与上述类别无关的一般性问题或无法明确分类的问题，则返回 "general"。
除了这几个选项之外，不要返回其他任何内容,并且不要添加任何额外的解释或文本。
"""

class RouterAgent:
    def __init__(self):
        self.llm = get_llm()

    async def run(self, state):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["input_message"]}
        ]

        response = await self.llm.invoke(messages)
        intent = response.content.strip().lower()
        if intent not in IntentType:
            intent = "general"

        state["intent"] = intent




