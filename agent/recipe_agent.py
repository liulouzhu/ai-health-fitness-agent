from agent.llm import get_llm
from agent.state import AgentState
from agent.memory_agent import get_memory_agent
from agent.context_manager import get_context_manager
from tools.search_with_tavily import search_with_tavily
from tools.retriever import get_hybrid_retriever


class RecipeAgent:
    def __init__(self):
        self.llm = get_llm()
        self.memory_agent = get_memory_agent()
        self.collection_name = "recipes"
        self._hybrid_retriever = None

    @property
    def hybrid_retriever(self):
        if self._hybrid_retriever is None:
            self._hybrid_retriever = get_hybrid_retriever(self.collection_name)
        return self._hybrid_retriever

    def _extract_constraints_from_input(self, user_input: str) -> str:
        """从用户输入中提取检索约束词（餐次、食材、口味、烹饪方式等）"""
        constraints = []
        meal_times = ["早餐", "午饭", "午餐", "晚饭", "晚餐", "宵夜", "夜宵", "加餐", "点心"]
        for m in meal_times:
            if m in user_input:
                constraints.append(m)
        ingredients = ["鸡胸肉", "牛肉", "鱼肉", "虾", "鸡蛋", "豆腐", "蔬菜", "西兰花", "菠菜",
                       "米饭", "面条", "面包", "红薯", "土豆", "藜麦", "牛油果", "坚果"]
        for ing in ingredients:
            if ing in user_input:
                constraints.append(ing)
        tastes = ["不要辣", "不辣", "微辣", "少油", "清淡", "重口", "麻辣", "咖喱", "蒜香"]
        for t in tastes:
            if t in user_input:
                constraints.append(t)
        methods = ["蒸", "煮", "炒", "烤", "煎", "炸", "拌", "炖", "快手", "简单", "容易", "做得快"]
        for m in methods:
            if m in user_input:
                constraints.append(m)
        return "、".join(constraints)

    def retrieve_from_qdrant(self, query: str, top_k: int = 5) -> list:
        """从本地向量数据库检索食谱（混合搜索：向量 + BM25）"""
        try:
            self.hybrid_retriever.fusion_top_k = top_k
            results = self.hybrid_retriever.retrieve(query)
            return results
        except Exception as e:
            print(f"食谱检索失败: {e}")
            return []

    def _is_retrieval_sufficient(self, retrieved_content: str) -> bool:
        """判断检索内容是否足够"""
        if not retrieved_content:
            return False
        judge_prompt = (
            f"判断以下检索内容是否足够推荐食谱。\n\n检索内容：{retrieved_content[:2000]}\n\n"
            f"如果检索内容足够，返回\"足够\"。如果不足，返回\"不足\"。\n只返回\"足够\"或\"不足\"。"
        )
        response = self.llm.invoke([{"role": "user", "content": judge_prompt}])
        return "足够" in response.content

    def _build_retrieval_query(self, user_input: str, ctx: dict) -> str:
        """构建检索 query：融合用户需求 + 营养约束 + 目标"""
        user_constraints = self._extract_constraints_from_input(user_input)
        constraints_str = f"，{user_constraints}" if user_constraints else ""
        return (
            f"{user_input}，"
            f"剩余{ctx['remaining_calories']}卡路里，"
            f"{ctx['remaining_protein']}克蛋白质，"
            f"健身{ctx['goal']}"
            f"{constraints_str}"
        )

    def run(self, state: AgentState) -> AgentState:
        """执行食谱推荐"""
        print(f"[RecipeAgent] run - 开始食谱推荐")
        try:
            ctx_mgr = get_context_manager()
            user_input = state.get("input_message", "")

            # 1. 通过 ContextManager 获取业务上下文（用于构建检索 query）
            bundle = ctx_mgr.build_context("recipe", state)
            task = bundle["task_context"]
            ctx = {
                "remaining_calories": task.get("remaining_calories", 0),
                "remaining_protein": task.get("remaining_protein", 0),
                "goal": task.get("goal", "维持"),
            }

            # 2. 构建检索 query 并检索
            query = self._build_retrieval_query(user_input, ctx)
            retrieved_results = self.retrieve_from_qdrant(query)
            retrieved_content = "\n".join([r.text for r in retrieved_results]) if retrieved_results else ""

            # 3. 判断检索是否足够，不足则联网补充
            if not self._is_retrieval_sufficient(retrieved_content):
                tavily_content = search_with_tavily(query)
                if tavily_content:
                    retrieved_content = f"{retrieved_content}\n\n--- 网络搜索结果 ---\n{tavily_content}"

            # 4. 通过 ContextManager 统一构建消息（含 token 预算管理）
            preferences = ctx_mgr.get_preferences_str()
            if not preferences:
                preferences = "（暂无偏好记录）"

            extra_sections = {
                "用户营养约束": (
                    f"剩余热量：{ctx['remaining_calories']} kcal；"
                    f"剩余蛋白质：{ctx['remaining_protein']} g；"
                    f"健身目标：{ctx['goal']}"
                ),
                "用户偏好": preferences,
                "参考食谱": retrieved_content or "无相关食谱",
            }
            messages = ctx_mgr.build_prompt_messages(
                "recipe",
                state,
                retrieved_content=retrieved_content,
                extra_sections=extra_sections,
            )

            response = self.llm.invoke(messages)
            state["response"] = response.content

        except Exception as e:
            print(f"[RecipeAgent] 错误: {e}")
            state["response"] = "抱歉，食谱推荐服务暂时不可用，请稍后重试。"

        # 更新对话历史（使用 ContextManager 统一管理滑动窗口）
        ctx_mgr.append_messages(state, state["input_message"], state.get("response", ""))
        return state
