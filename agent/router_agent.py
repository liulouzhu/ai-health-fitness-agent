from agent.llm import get_llm
from agent.state import AgentState
from agent.memory_agent import get_memory_agent
from agent.context_manager import get_context_manager

INTENT_TYPES = ["food", "workout", "recipe", "stats_query", "profile_update", "confirm", "general", "food_report", "workout_report"]

CONFIRM_WORDS = ["是", "好的", "记录", "yes", "确认", "计入", "算", "嗯", "ok", "okay"]
DENY_WORDS = ["no", "取消", "不算", "不计入"]


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
        try:
            print(f"[Router] classify_intent - 用户输入: {state.get('input_message', '')[:50]}...")

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
            # 使用 ContextManager 统一构建（保证窗口裁剪逻辑一致）
            ctx_mgr = get_context_manager()
            messages = ctx_mgr.build_prompt_messages(
                "classify_intent",
                state,
            )
            print(f"[Router] classify_intent - 对话历史 messages 数量: {len(state.get('messages', []))}")

            response = self.llm.invoke(messages)
            raw_intent = response.content.strip().lower()

            # 解析多意图（以逗号分隔）
            intent_list = [i.strip() for i in raw_intent.split(",") if i.strip()]
            # 过滤掉无效意图，保留有效的
            valid_intents = [i for i in intent_list if i in INTENT_TYPES]

            if not valid_intents:
                valid_intents = ["general"]

            # 判断是否是用户主动报告吃食物或做运动
            is_reporting = self._is_user_reporting_food_or_workout(user_input)

            # 如果是主动报告且意图是 food/workout，修改意图类型
            if is_reporting:
                if valid_intents[0] == "food":
                    valid_intents = ["food_report"] + valid_intents[1:]
                elif valid_intents[0] == "workout":
                    valid_intents = ["workout_report"] + valid_intents[1:]

            # 保留第一个作为主意图（兼容现有逻辑）
            state["intent"] = valid_intents[0]
            # 保存多意图列表
            state["intents"] = valid_intents
            # 保存上一个意图
            state["last_intent"] = valid_intents[0]
            print(f"[Router] classify_intent - 分类结果: {valid_intents}, last_intent: {valid_intents[0]}")
        except Exception as e:
            print(f"[Router] classify_intent 错误: {e}")
            state["intent"] = "general"
            state["last_intent"] = None
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
        return len(user_input) <= 10 and any(word in user_input for word in CONFIRM_WORDS + DENY_WORDS)

    def _is_user_reporting_food_or_workout(self, text: str) -> bool:
        """判断用户是否在主动报告吃食物或做运动"""
        if not text:
            return False
        food_report_words = ["吃了", "吃了点", "吃了碗", "吃了份", "吃了些", "摄入了", "吃进去", "吃了一个", "吃了俩", "干掉", "干饭", "喝了"]
        workout_report_words = ["跑了", "跑了步", "做了", "做了运动", "健身了", "练了", "锻炼了", "运动了", "跑步了", "游泳了", "骑车了", "走路了", "跳绳了", "打球了", "健身"]
        text_lower = text.lower()
        return any(word in text_lower for word in food_report_words + workout_report_words)

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

        # 判断是肯定还是否定（优先检查否定词，防止"不算"/"不计入"等部分匹配）
        is_deny = any(word == user_input for word in DENY_WORDS)
        is_yes = not is_deny and any(word in user_input for word in CONFIRM_WORDS)

        if is_yes:
            # 保存统计（支持 multi 类型：同时保存 food 和 workout）
            if pending.get("type") == "multi":
                if pending.get("food"):
                    self.memory_agent.update_daily_stats("meal", pending["food"])
                if pending.get("workout"):
                    self.memory_agent.update_daily_stats("workout", pending["workout"])
                state["response"] = "已计入统计（食物+运动）。\n\n" + self.memory_agent.get_daily_summary()
                print(f"[Router] handle_confirm - 用户确认，已计入 food + workout")
            else:
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
        try:
            if not state.get("profile_complete", True):
                state["response"] = self.memory_agent.get_initial_questions()
                return state

            # 通过 ContextManager 统一构建消息列表
            ctx_mgr = get_context_manager()
            messages = ctx_mgr.build_prompt_messages("general", state)

            # 使用 stream() 获取流式响应
            stream_resp = self.llm.stream(messages)
            # 收集完整响应
            full_response = ""
            for chunk in stream_resp:
                if chunk.content:
                    full_response += chunk.content

            state["response"] = full_response

            # 更新对话历史（使用 ContextManager 统一管理滑动窗口）
            ctx_mgr.append_messages(state, state["input_message"], full_response)
            print(f"[Router] handle_general - 更新后 messages 数量: {len(state.get('messages', []))}")
        except Exception as e:
            print(f"[Router] handle_general 错误: {e}")
            state["response"] = "抱歉，服务暂时不可用，请稍后重试。"
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
