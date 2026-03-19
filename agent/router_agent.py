from agent.llm import get_llm
from agent.state import AgentState
from agent.memory_agent import get_memory_agent

INTENT_TYPES = ["food", "workout", "recipe", "stats_query", "profile_update", "confirm", "general"]

SYSTEM_PROMPT = """你是一个智能路由器，负责判断用户意图并路由到对应的Agent。

可选意图：
- "food": 食物、餐饮、卡路里、营养成分、图片中的食物识别
- "workout": 锻炼、健身计划、运动建议、训练方法、热量消耗统计
- "recipe": 食谱推荐、推荐晚餐/午餐/早餐、吃什么好、想吃点啥
- "stats_query": 查询今日热量消耗、今日吃了多少、还剩多少热量、今天统计
- "profile_update": 用户主动更新档案信息（如"我体重变成XX了"、"我长高了"等）
- "confirm": 用户回复"是/否"用于确认之前的操作
- "general": 其他一般性对话、问候、无法分类的问题

判断规则：
1. 如果用户发送了图片（无论文字说什么），优先判断为 "food"（食物识别）
2. 如果用户提到体重、身高、年龄、性别、目标变化，判断为 "profile_update"
3. 如果用户回复"是/否/确认/取消"等简短回答，判断为 "confirm"
4. 如果用户问"今天吃了多少"、"还剩多少热量"、"今日统计"、"热量消耗"，判断为 "stats_query"
5. 如果用户问"吃什么"、"推荐食物"、"食谱"、"早/午/晚餐推荐"，判断为 "recipe"
6. 如果用户问"怎么练"、"动作要领"，判断为 "workout"
7. 如果用户提到运动但没有具体问题（如"我跑步了"、"游泳了一小时"），判断为 "workout"

只返回意图标签，不要任何解释。"""

GENERAL_PROMPT = """你是一个健身健康智能助手，可以进行日常对话。

对话历史：
{history}

用户：{input}

请回复用户，保持对话连贯性。如果用户问到健身或饮食相关问题，可以适当引导。"""


class RouterAgent:
    def __init__(self):
        self.llm = get_llm()
        self.memory_agent = get_memory_agent()

    def check_profile(self, state: AgentState) -> AgentState:
        """检查用户档案是否完整"""
        print(f"[Router] check_profile - 检查用户档案是否完整")
        if self.memory_agent.is_profile_complete():
            state["profile_complete"] = True
            print(f"[Router] check_profile - 档案完整")
        else:
            state["profile_complete"] = False
            state["response"] = self.memory_agent.get_initial_questions()
            print(f"[Router] check_profile - 档案不完整，请求用户填写")
        return state

    def classify_intent(self, state: AgentState) -> AgentState:
        """意图分类node"""
        print(f"[Router] classify_intent - 开始意图分类")
        user_input = state.get("input_message", "")
        print(f"[Router] classify_intent - 用户输入: {user_input[:50]}...")

        # 如果档案不完整，先处理档案创建/更新
        if not state.get("profile_complete", True):
            user_input = state.get("input_message", "")
            if any(c.isdigit() for c in user_input):
                state["intent"] = "profile_update"
            else:
                state["intent"] = "general"
            return state

        image_info = state.get("image_info", {})
        if image_info.get("has_image", False):
            state["intent"] = "food"
            return state

        # 检查是否是确认回答
        user_input = state.get("input_message", "").strip().lower()
        if self._is_confirmation(user_input):
            state["intent"] = "confirm"
            return state

        # 检查是否是上下文相关的短回复（如"换一个"、"再来一个"等）
        context_intent = self._check_context_dependent_intent(state)
        if context_intent:
            state["intent"] = context_intent
            state["last_intent"] = context_intent
            return state

        # 构建消息列表，融入对话历史
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 添加对话历史
        history = state.get("messages", [])
        for msg in history[-10:]:  # 只取最近10条历史
            messages.append(msg)

        # 添加当前输入
        messages.append({"role": "user", "content": state["input_message"]})

        response = self.llm.invoke(messages)
        intent = response.content.strip().lower()

        if intent not in INTENT_TYPES:
            intent = "general"

        state["intent"] = intent
        # 保存上一个意图
        state["last_intent"] = intent
        print(f"[Router] classify_intent - 分类结果: {intent}, last_intent: {intent}")
        return state

    def _check_context_dependent_intent(self, state: AgentState) -> str:
        """检查是否是上下文相关的意图（如紧跟食谱请求后的"换一个"）"""
        user_input = state.get("input_message", "").strip().lower()
        last_intent = state.get("last_intent")

        # 上下文相关的跟随词
        follow_up_words = ["换一个", "再来一个", "其他", "换个", "换一", "换种", "换道", "换个口", "重新", "再来"]

        # 如果用户没有使用跟随词，不做上下文推断
        if not any(word in user_input for word in follow_up_words):
            return None

        print(f"[Router] _check_context_dependent_intent - 检测到跟随词，上一个意图: {last_intent}")

        # 如果上一条是食谱/运动相关，用户说跟随词 -> 仍然是 recipe/workout
        if last_intent == "recipe":
            print(f"[Router] _check_context_dependent_intent - 继承 recipe 意图")
            return "recipe"
        if last_intent == "workout":
            print(f"[Router] _check_context_dependent_intent - 继承 workout 意图")
            return "workout"

        return None

    def _is_confirmation(self, user_input: str) -> bool:
        """判断是否是确认回答"""
        if not user_input:
            return False
        confirm_words = ["是", "否", "yes", "no", "确认", "取消", "算", "不算", "计入", "不计入"]
        return len(user_input) <= 5 and any(word in user_input for word in confirm_words)

    def routing_func(self, state: AgentState) -> str:
        """条件路由函数"""
        intent = state.get("intent", "general")
        print(f"[Router] routing_func - 路由到: {intent}")
        return intent

    def handle_profile_update(self, state: AgentState) -> AgentState:
        """处理档案更新"""
        print(f"[Router] handle_profile_update - 处理档案更新")
        user_input = state.get("input_message", "")

        if not self.memory_agent.load_profile().get("height"):
            try:
                profile = self.memory_agent.create_profile(user_input)
                state["user_profile"] = profile
                state["response"] = f"档案创建成功！\n\n{self._format_profile(profile)}"
            except Exception as e:
                state["response"] = f"抱歉，无法解析你的回答，请重新描述：{e}"
        else:
            result = self.memory_agent.update_profile(user_input)
            if result.get("changed"):
                profile = self.memory_agent.load_profile()
                state["user_profile"] = profile
                state["response"] = f"档案已更新！\n\n{self._format_profile(profile)}"
            else:
                state["response"] = result.get("message", "未检测到档案变化")

        return state

    def handle_confirm(self, state: AgentState) -> AgentState:
        """处理确认回答"""
        print(f"[Router] handle_confirm - 处理确认")
        user_input = state.get("input_message", "").strip().lower()

        # 优先从临时文件加载待确认数据
        pending = self.memory_agent.load_pending_stats()

        if not pending:
            # 没有待确认的数据
            state["response"] = "没有待确认的记录，请先查询食物或运动信息。"
            return state

        # 判断是肯定还是否定
        is_yes = any(word in user_input for word in ["是", "yes", "确认", "算", "计入"])

        if is_yes:
            # 保存统计
            self.memory_agent.update_daily_stats(pending["type"], pending["data"])
            summary = self.memory_agent.get_daily_summary()
            state["response"] = f"已计入统计。\n\n{summary}"
            print(f"[Router] handle_confirm - 用户确认，已计入统计")
        else:
            # 取消
            state["response"] = f"好的，已取消。"
            print(f"[Router] handle_confirm - 用户取消")

        # 清除待确认数据
        self.memory_agent.clear_pending_stats()
        state["pending_stats"] = None
        return state

    def handle_general(self, state: AgentState) -> AgentState:
        """一般对话node"""
        print(f"[Router] handle_general - 处理一般对话")
        if not state.get("profile_complete", True):
            state["response"] = self.memory_agent.get_initial_questions()
            return state

        # 获取现有对话历史
        messages = state.get("messages", [])

        # 构建对话历史字符串（只取最近10条）
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[-10:]])

        prompt = GENERAL_PROMPT.format(
            history=history_str or "（无历史对话）",
            input=state["input_message"]
        )

        # 构建发送给LLM的消息
        llm_messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": state["input_message"]}
        ]

        response = self.llm.invoke(llm_messages)

        state["response"] = response.content

        # 将本次对话添加到历史
        state["messages"] = messages + [
            {"role": "user", "content": state["input_message"]},
            {"role": "assistant", "content": response.content}
        ]

        return state

    def handle_stats_query(self, state: AgentState) -> AgentState:
        """处理统计查询"""
        print(f"[Router] handle_stats_query - 处理统计查询")
        summary = self.memory_agent.get_daily_summary()
        state["response"] = summary
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
