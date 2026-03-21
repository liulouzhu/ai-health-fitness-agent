from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel
from agent.llm import get_llm
from agent.state import AgentState
from agent.memory_agent import get_memory_agent

FOOD_AGENT_PROMPT = """你是一个食物营养分析专家。请分析用户询问的食物并提供营养信息。

用户偏好（请在推荐和分析时注意）：
{preferences}

分析内容：
- 食物名称
- 热量 (kcal)
- 蛋白质 (g)
- 脂肪 (g)
- 碳水化合物 (g)

直接回复分析结果，不需要额外解释。"""


class NutritionInfo(BaseModel):
    """营养信息结构"""
    name: str
    calories: float
    protein: float
    fat: float
    carbs: float


@tool
def extract_nutrition(food_description: str) -> NutritionInfo:
    """从食物描述中提取营养信息"""
    return NutritionInfo(
        name=food_description,
        calories=0,
        protein=0,
        fat=0,
        carbs=0
    )


class FoodAgent:
    def __init__(self):
        self.llm = get_llm()
        self.memory_agent = get_memory_agent()
        self.llm_with_tools = self.llm.bind_tools([extract_nutrition])

    def _extract_nutrition(self, food_description: str) -> dict:
        """从食物描述中提取营养数据"""
        try:
            response = self.llm_with_tools.invoke([
                {"role": "user", "content": f"从以下文本中提取营养信息：{food_description}"}
            ])

            if response.tool_calls:
                parsed = response.tool_calls[0].parsed
                return {
                    "name": parsed.name,
                    "calories": parsed.calories,
                    "protein": parsed.protein,
                    "fat": parsed.fat,
                    "carbs": parsed.carbs
                }
        except Exception as e:
            print(f"[FoodAgent] 解析营养数据失败: {e}")

        return {
            "name": "未知食物",
            "calories": 0,
            "protein": 0,
            "fat": 0,
            "carbs": 0
        }

    def run(self, state: AgentState) -> AgentState:
        """执行食物分析"""
        print(f"[FoodAgent] run - 开始分析食物")
        try:
            image_info = state.get("image_info", {})

            # 获取用户偏好
            preferences = self.memory_agent.get_preferences_for_context()
            if not preferences:
                preferences = "（暂无偏好记录）"

            # 注入偏好的 prompt
            prompt_with_prefs = FOOD_AGENT_PROMPT.format(preferences=preferences)

            if image_info.get("has_image", False):
                # 带图片的输入
                image_url = image_info.get("image_url", "")
                messages = [
                    {"role": "system", "content": prompt_with_prefs},
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
                    {"role": "system", "content": prompt_with_prefs},
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

            # 设置待确认数据
            state["pending_stats"] = {
                "type": "meal",
                "data": entry,
                "response": response.content
            }
            self.memory_agent.save_pending_stats(state["pending_stats"])

            # 设置响应（单意图时直接返回，多意图时由 multi_intent_node 合并）
            state["response"] = f"{response.content}\n\n---\n是否将上述食物计入今日热量统计？（是/否）"

        except Exception as e:
            print(f"[FoodAgent] 错误: {e}")
            state["response"] = "抱歉，食物分析服务暂时不可用，请稍后重试。"
            state["food_result"] = None
            state["pending_stats"] = None
        return state
