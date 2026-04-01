"""
食物分析 Agent

职责（graph-native 改造后）：
- 输入解析与营养数据提取
- 生成候选记录 candidate_meal
- 设置 pending_action / requires_confirmation
- 由 graph 的 confirm_node → commit_node 完成实际写入

不再直接操作 memory_agent 的 save_pending_stats / update_daily_stats，
这些统一收敛到 graph 的 commit_node。
"""

from langchain_core.messages import HumanMessage
from agent.llm import get_llm
from agent.state import AgentState
from agent.memory_agent import get_memory_agent
from agent.context_manager import get_context_manager, is_retrieval_sufficient as check_retrieval
from tools.search_with_tavily import search_with_tavily
from tools.retriever import get_hybrid_retriever


class FoodAgent:
    def __init__(self):
        self.llm = get_llm()
        self.memory_agent = get_memory_agent()
        self.collection_name = "fitness_guide"
        self._hybrid_retriever = None
        self.llm_with_tools = self.llm.bind_tools([FoodInfoTool])

    @property
    def hybrid_retriever(self):
        if self._hybrid_retriever is None:
            self._hybrid_retriever = get_hybrid_retriever(self.collection_name)
        return self._hybrid_retriever

    def retrieve_from_qdrant(self, query: str, top_k: int = 3) -> list:
        """从本地向量数据库检索（混合搜索：向量 + BM25）"""
        try:
            self.hybrid_retriever.fusion_top_k = top_k
            results = self.hybrid_retriever.retrieve(query)
            return results
        except Exception as e:
            print(f"[FoodAgent] Qdrant检索失败: {e}")
            return []

    def _extract_nutrition(self, food_description: str) -> dict:
        """从 LLM 输出中提取营养数据"""
        try:
            prompt = f"""从以下食物描述中提取营养信息，返回JSON格式：
食物描述：{food_description}

返回格式（仅返回JSON，不要其他文字）：
{{"name": "食物名称", "calories": 热量数值, "protein": 蛋白质克数, "fat": 脂肪克数, "carbs": 碳水化合物克数}}"""

            response = self.llm.invoke([{"role": "user", "content": prompt}])
            print(f"[FoodAgent] _extract_nutrition - response.content: {response.content}")
            data = _extract_json_from_llm(response.content)
            if data:
                return {
                    "name": data.get("name", "未知食物"),
                    "calories": float(data.get("calories", 0) or 0),
                    "protein": float(data.get("protein", 0) or 0),
                    "fat": float(data.get("fat", 0) or 0),
                    "carbs": float(data.get("carbs", 0) or 0)
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
        """执行食物分析（graph-native 版本）

        流程：
        1. 解析输入（图片或文字）
        2. 检索/生成营养数据
        3. 生成候选记录 candidate_meal
        4. 设置 pending_action + requires_confirmation
        5. 写入 food_result + final_response

        实际 commit 由 graph 的 commit_node 负责。
        """
        is_user_reporting = state.get("intent") == "food_report"
        print(f"[FoodAgent] run - is_user_reporting={is_user_reporting}, intent={state.get('intent')}")

        try:
            image_info = state.get("image_info", {})
            ctx_mgr = get_context_manager()
            query = state["input_message"]

            # === Step 1: 检索 ===
            retrieved_results = self.retrieve_from_qdrant(query)
            retrieved_content = "\n".join([r.text for r in retrieved_results]) if retrieved_results else ""

            if not check_retrieval(retrieved_content, query=query, domain="food"):
                tavily_content = search_with_tavily(query)
                if tavily_content:
                    retrieved_content = f"{retrieved_content}\n\n--- 网络搜索结果 ---\n{tavily_content}"

            # === Step 2: 获取偏好和上下文 ===
            preferences = ctx_mgr.get_preferences_str()
            if not preferences:
                preferences = "（暂无偏好记录）"
            task_text = ctx_mgr.format_task_context("food")

            # === Step 3: LLM 分析 ===
            if image_info.get("has_image", False):
                # 多模态分支
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
                user_text = f"用户问题：{query}\n请分析这张图片中的食物营养成分。" if query else "请分析这张图片中的食物营养成分。"
                messages = [
                    {"role": "system", "content": system_prompt},
                    HumanMessage(
                        content=[
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    )
                ]
                response = self.llm.invoke(messages)
                state["food_result"] = response.content
                nutrition = self._extract_nutrition(response.content)
            else:
                # 文字分支：DB 主查 + LLM 兜底
                from tools.food_service import get_food_service
                food_service = get_food_service()
                nutrition, had_macros = food_service.lookup(query)

                if had_macros:
                    # DB 已有完整宏量营养素
                    state["food_result"] = (
                        f"{nutrition['name']}："
                        f"热量 {nutrition['calories']} kcal，"
                        f"蛋白质 {nutrition['protein']}g，"
                        f"脂肪 {nutrition['fat']}g，"
                        f"碳水 {nutrition['carbs']}g。"
                    )
                else:
                    # DB miss 或缺宏量营养素 → LLM 生成
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
                    state["food_result"] = response.content
                    # 用 LLM 提取的营养数据覆盖（可能更完整）
                    llm_nutrition = self._extract_nutrition(response.content)
                    if llm_nutrition.get("protein", 0) > 0 or llm_nutrition.get("fat", 0) > 0:
                        nutrition = llm_nutrition

            # === Step 4: 构造候选记录 ===
            entry = {
                "name": nutrition.get("name", query[:20]),
                "calories": nutrition.get("calories", 0),
                "protein": nutrition.get("protein", 0),
                "fat": nutrition.get("fat", 0),
                "carbs": nutrition.get("carbs", 0)
            }
            state["candidate_meal"] = entry

            # === Step 5: 设置 pending_action / requires_confirmation ===
            # 主动报告（food_report）：confirmed=True → graph 直接路由到 commit_node 写入
            # 查询（food）：confirmed=None → graph 路由到 confirm_node 询问用户
            if is_user_reporting:
                state["pending_action"] = "log_meal"
                state["requires_confirmation"] = True
                state["pending_confirmation"] = {
                    "action": "log_meal",
                    "candidate_meal": entry,
                    "candidate_workout": None,
                    "analysis_text": state["food_result"],
                    "confirmed": True,  # confirmed=True → commit_node 直接 commit
                }
                state["final_response"] = (
                    f"{state['food_result']}\n\n---\n"
                    f"（已提交，等待确认...）"
                )
                state["response"] = state["final_response"]
            else:
                state["pending_action"] = "log_meal"
                state["requires_confirmation"] = True
                state["pending_confirmation"] = {
                    "action": "log_meal",
                    "candidate_meal": entry,
                    "candidate_workout": None,
                    "analysis_text": state["food_result"],
                    "confirmed": None,  # confirmed=None → confirm_node 询问用户
                }
                # confirm_node 会覆盖 final_response 为确认提示，所以这里只写 food_result
                state["final_response"] = state["food_result"]
                state["response"] = state["final_response"]

        except Exception as e:
            import traceback
            print(f"[FoodAgent] 错误: {e}")
            traceback.print_exc()
            state["response"] = "抱歉，食物分析服务暂时不可用，请稍后重试。"
            state["food_result"] = None
            state["candidate_meal"] = None
            state["pending_action"] = None
            state["requires_confirmation"] = False

        # 更新对话历史
        ctx_mgr = get_context_manager()
        ctx_mgr.append_messages(state, state["input_message"], state.get("response", ""))
        return state


class FoodInfoTool:
    """食物信息工具 schema（供 LLM tool calling 使用）"""
    name = "FoodInfo"
    description = "提取食物营养信息"
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "食物名称"},
            "calories": {"type": "number", "description": "热量(kcal)"},
            "protein": {"type": "number", "description": "蛋白质(g)"},
            "fat": {"type": "number", "description": "脂肪(g)"},
            "carbs": {"type": "number", "description": "碳水化合物(g)"},
        },
        "required": ["name", "calories"]
    }


def _extract_json_from_llm(text: str) -> dict:
    """从 LLM 输出中提取 JSON"""
    import json, re
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试从 markdown code block 中提取
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # 尝试提取 {...} 格式
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}
