from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel
from agent.llm import get_llm
from agent.state import AgentState
from agent.memory_agent import get_memory_agent
from agent.context_manager import get_context_manager


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
            prompt = f"""从以下食物描述中提取营养信息，返回JSON格式：
食物描述：{food_description}

返回格式（仅返回JSON，不要其他文字）：
{{"name": "食物名称", "calories": 热量数值, "protein": 蛋白质克数, "fat": 脂肪克数, "carbs": 碳水化合物克数}}"""

            response = self.llm.invoke([
                {"role": "user", "content": prompt}
            ])

            print(f"[FoodAgent] _extract_nutrition - response.content: {response.content}")

            import json, re
            json_match = re.search(r'\{[^}]+\}', response.content)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "name": data.get("name", "未知食物"),
                    "calories": float(data.get("calories", 0) or 0),
                    "protein": float(data.get("protein", 0) or 0),
                    "fat": float(data.get("fat", 0) or 0),
                    "carbs": float(data.get("carbs", 0) or 0)
                }
        except Exception as e:
            print(f"[FoodAgent] 解析营养数据失败: {e}")
            print(f"[FoodAgent] 原始响应: {response.content if 'response' in dir() else 'N/A'}")

        return {
            "name": "未知食物",
            "calories": 0,
            "protein": 0,
            "fat": 0,
            "carbs": 0
        }

    def run(self, state: AgentState) -> AgentState:
        """执行食物分析"""
        is_user_reporting = state.get("intent") == "food_report"
        print(f"[FoodAgent] run - is_user_reporting={is_user_reporting}, intent={state.get('intent')}")
        try:
            image_info = state.get("image_info", {})
            ctx_mgr = get_context_manager()

            # 通过公开 API 获取偏好
            preferences = ctx_mgr.get_preferences_str()
            if not preferences:
                preferences = "（暂无偏好记录）"

            # 通过统一上下文接口获取 task_context 格式化文本
            task_text = ctx_mgr.format_task_context("food")

            if image_info.get("has_image", False):
                # 带图片的多模态输入：system prompt 直接注入偏好和 task_context
                system_parts = [
                    "你是一个食物营养分析专家。请分析用户询问的食物并提供营养信息。\n",
                    f"用户偏好：{preferences}",
                ]
                if task_text:
                    system_parts.append(f"今日情况：{task_text}")
                system_parts.append(
                    "\n分析内容：食物名称、热量(kcal)、蛋白质(g)、脂肪(g)、碳水化合物(g)。"
                    "\n直接回复分析结果，不需要额外解释。"
                )
                system_prompt = "\n".join(system_parts)

                image_url = image_info.get("image_url", "")
                messages = [
                    {"role": "system", "content": system_prompt},
                    HumanMessage(
                        content=[
                            {"type": "text", "text": "请分析这张图片中的食物营养成分。"},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    )
                ]
            else:
                # 纯文字：使用 ContextManager 统一构建（含 token 预算管理）
                extra_sections = {
                    "用户偏好": preferences,
                    "今日情况": task_text,
                }
                messages = ctx_mgr.build_prompt_messages(
                    "food",
                    state,
                    extra_sections=extra_sections,
                )

            response = self.llm.invoke(messages)
            print(f"[FoodAgent] LLM响应长度: {len(response.content) if response.content else 0}")
            state["food_result"] = response.content

            # 提取营养数据
            print(f"[FoodAgent] 开始提取营养数据...")
            nutrition = self._extract_nutrition(response.content)
            print(f"[FoodAgent] 营养数据: {nutrition}")
            entry = {
                "name": nutrition.get("name", state["input_message"][:20]),
                "calories": nutrition.get("calories", 0),
                "protein": nutrition.get("protein", 0),
                "fat": nutrition.get("fat", 0),
                "carbs": nutrition.get("carbs", 0)
            }

            # 判断是否是用户主动报告（直接记录）还是询问（需要确认）
            if is_user_reporting:
                print(f"[FoodAgent] 开始保存营养数据: {entry}")
                self.memory_agent.update_daily_stats("meal", entry)
                print(f"[FoodAgent] 保存完成，获取每日摘要...")
                summary = self.memory_agent.get_daily_summary()
                print(f"[FoodAgent] 摘要长度: {len(summary) if summary else 0}")
                state["response"] = f"{response.content}\n\n---\n已记录。\n\n{summary}"
                state["pending_stats"] = None
                self.memory_agent.clear_pending_stats()
            else:
                state["pending_stats"] = {
                    "type": "meal",
                    "data": entry,
                    "response": response.content
                }
                self.memory_agent.save_pending_stats(state["pending_stats"])
                state["response"] = f"{response.content}\n\n---\n是否将上述食物计入今日热量统计？（是/否）"

        except Exception as e:
            import traceback
            print(f"[FoodAgent] 错误: {e}")
            traceback.print_exc()
            state["response"] = "抱歉，食物分析服务暂时不可用，请稍后重试。"
            state["food_result"] = None
            state["pending_stats"] = None

        # 更新对话历史（使用 ContextManager 统一管理滑动窗口）
        ctx_mgr.append_messages(state, state["input_message"], state.get("response", ""))
        return state
