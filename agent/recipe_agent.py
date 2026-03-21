from agent.llm import get_llm, get_embedding_model
from config import AgentConfig
from agent.state import AgentState
from agent.memory_agent import get_memory_agent
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from tools.search_with_tavily import search_with_tavily

RECIPE_AGENT_PROMPT = """你是一个营养师。根据用户的饮食目标和限制，推荐合适的食谱。

用户信息：
- 剩余热量：{remaining_calories} kcal
- 剩余蛋白质：{remaining_protein} g
- 健身目标：{goal}

用户偏好（请在推荐时严格遵守）：
{preferences}

参考食谱：
{retrieved_recipes}

请根据以上信息，推荐合适的食谱组合，确保：
1. 总热量不超过剩余热量
2. 蛋白质尽量达到目标
3. 食物种类多样化
4. **严格避免推荐用户不喜欢或过敏的食物**

直接回复推荐内容，不需要额外解释。"""

JUDGE_RECIPE_PROMPT = """判断以下检索内容是否足够推荐食谱。

检索内容：
{retrieved_content}

如果检索内容足够，返回"足够"。
如果不足，返回"不足"。

只返回"足够"或"不足"。"""


class RecipeAgent:
    def __init__(self):
        self.llm = get_llm()
        self.embed_model = get_embedding_model()
        self.memory_agent = get_memory_agent()
        self.qdrant_client = QdrantClient(host=AgentConfig.QDRANT_HOST, port=AgentConfig.QDRANT_PORT)
        self.collection_name = "recipes"
        self._vector_store = None

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embed_model
            )
        return self._vector_store

    def retrieve_from_qdrant(self, query: str, top_k: int = 5) -> list:
        """从本地向量数据库检索食谱"""
        try:
            index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model
            )
            retriever = index.as_retriever(similarity_top_k=top_k)
            results = retriever.retrieve(query)
            return results
        except Exception as e:
            print(f"食谱检索失败: {e}")
            return []

    def is_retrieval_sufficient(self, retrieved_content: str) -> bool:
        """判断检索内容是否足够"""
        if not retrieved_content:
            return False

        judge_prompt = JUDGE_RECIPE_PROMPT.format(
            retrieved_content=retrieved_content[:AgentConfig.RETRIEVAL_CONTENT_TRUNCATE]
        )

        response = self.llm.invoke([{"role": "user", "content": judge_prompt}])
        return "足够" in response.content

    def get_recommendation_context(self) -> dict:
        """获取推荐所需的上下文信息"""
        # 获取用户档案（长期记忆）
        profile = self.memory_agent.load_profile()
        target_cal = int(profile.get("target_calories", 2000))
        target_pro = int(profile.get("target_protein", 100))
        goal = profile.get("goal", "维持")
        goal_names = {"cut": "减脂", "bulk": "增肌", "maintain": "维持"}
        goal_display = goal_names.get(goal, goal)

        # 获取今日统计（短期记忆）
        today = self.memory_agent.load_daily_stats()
        consumed_cal = today.get("consumed_calories", 0)
        consumed_pro = today.get("consumed_protein", 0)
        burned_cal = today.get("burned_calories", 0)

        # 计算剩余
        remaining_cal = max(0, target_cal - consumed_cal + burned_cal)
        remaining_pro = max(0, target_pro - consumed_pro)

        # 获取用户偏好
        preferences = self.memory_agent.get_preferences_for_context()
        if not preferences:
            preferences = "（暂无偏好记录）"

        return {
            "remaining_calories": remaining_cal,
            "remaining_protein": remaining_pro,
            "goal": goal_display,
            "consumed_calories": consumed_cal,
            "consumed_protein": consumed_pro,
            "target_calories": target_cal,
            "target_protein": target_pro,
            "burned_calories": burned_cal,
            "preferences": preferences
        }

    def run(self, state: AgentState) -> AgentState:
        """执行食谱推荐"""
        print(f"[RecipeAgent] run - 开始食谱推荐")
        try:
            # 获取推荐上下文
            context = self.get_recommendation_context()

            # 构建检索query
            query = f"剩余{context['remaining_calories']}卡路里，{context['remaining_protein']}克蛋白质的食谱推荐，健身{context['goal']}"

            # 1. 首先从本地向量数据库检索
            retrieved_results = self.retrieve_from_qdrant(query)
            retrieved_content = "\n".join([r.text for r in retrieved_results]) if retrieved_results else ""

            # 2. 判断检索内容是否足够
            if not self.is_retrieval_sufficient(retrieved_content):
                # 3. 不足则使用Tavily搜索
                tavily_content = search_with_tavily(query)
                if tavily_content:
                    retrieved_content = f"{retrieved_content}\n\n--- 网络搜索结果 ---\n{tavily_content}"

            # 4. 组装提示并调用LLM
            prompt = RECIPE_AGENT_PROMPT.format(
                remaining_calories=context["remaining_calories"],
                remaining_protein=context["remaining_protein"],
                goal=context["goal"],
                retrieved_recipes=retrieved_content or "无相关食谱",
                preferences=context["preferences"]
            )

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": state["input_message"]}
            ]

            response = self.llm.invoke(messages)
            state["response"] = response.content
        except Exception as e:
            print(f"[RecipeAgent] 错误: {e}")
            state["response"] = "抱歉，食谱推荐服务暂时不可用，请稍后重试。"
        return state
