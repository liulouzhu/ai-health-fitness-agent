from langchain_core.messages import HumanMessage
from agent.llm import get_llm
from agent.state import AgentState
from agent.memory_agent import get_memory_agent

FOOD_AGENT_PROMPT = """你是一个食物营养分析专家。请分析用户询问的食物并提供营养信息。

分析内容：
- 食物名称
- 热量 (kcal)
- 蛋白质 (g)
- 脂肪 (g)
- 碳水化合物 (g)

直接回复分析结果，不需要额外解释。"""

EXTRACT_NUTRITION_PROMPT = """从以下食物分析结果中提取营养数据。

分析结果：
{analysis_result}

请以JSON格式返回：
{{"name": "食物名称", "calories": 数字, "protein": 数字, "fat": 数字, "carbs": 数字}}

如果分析结果中没有提供某项数据，该项设为0。"""


class FoodAgent:
    def __init__(self):
        self.llm = get_llm()
        self.memory_agent = get_memory_agent()

    def _extract_nutrition(self, analysis_result: str) -> dict:
        """从分析结果中提取营养数据"""
        prompt = EXTRACT_NUTRITION_PROMPT.format(analysis_result=analysis_result)
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        try:
            import json
            data = json.loads(response.content)
        except:
            data = {
                "name": "未知食物",
                "calories": 0,
                "protein": 0,
                "fat": 0,
                "carbs": 0
            }

        return data

    def run(self, state: AgentState) -> AgentState:
        """执行食物分析"""
        print(f"[FoodAgent] run - 开始分析食物")
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

        # 提取营养数据，设置待确认
        nutrition = self._extract_nutrition(response.content)
        entry = {
            "name": nutrition.get("name", state["input_message"][:20]),
            "calories": nutrition.get("calories", 0),
            "protein": nutrition.get("protein", 0),
            "fat": nutrition.get("fat", 0),
            "carbs": nutrition.get("carbs", 0)
        }

        # 设置待确认数据，交给 confirm_node 处理
        pending = {
            "type": "meal",
            "data": entry,
            "response": response.content
        }
        state["pending_stats"] = pending
        self.memory_agent.save_pending_stats(pending)

        # 返回分析结果 + 确认问题
        state["response"] = f"{response.content}\n\n---\n是否将上述食物计入今日热量统计？（是/否）"
        return state
