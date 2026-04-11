from agent.llm import get_llm
from agent.state import AgentState
from agent.memory import get_memory_agent
from agent.context_manager import get_context_manager
from agent.stream_utils import emit_trace, emit_event

INTENT_TYPES = ["food", "workout", "recipe", "stats_query", "profile_update", "confirm", "general", "food_report", "workout_report"]

CONFIRM_WORDS = ["是", "好的", "yes", "确认", "计入", "嗯", "ok", "okay"]
DENY_WORDS = ["no", "取消", "不算", "不计入"]

# 预计算的集合，用于快速查找
_CONFIRM_WORDS_SET = set(CONFIRM_WORDS)
_DENY_WORDS_SET = set(DENY_WORDS)


class RouterAgent:
    def __init__(self):
        self.llm = get_llm()
        self.memory_agent = get_memory_agent()

    def check_profile(self, state: AgentState) -> AgentState:
        """检查用户档案是否完整"""
        print(f"[Router] check_profile - 检查用户档案是否完整")
        emit_trace("node_start", "check_profile", "正在检查用户档案...")
        state["route_decision"] = "check_profile"
        if self.memory_agent.is_profile_complete():
            state["profile_complete"] = True
            print(f"[Router] check_profile - 档案完整")
        else:
            state["profile_complete"] = False
            state["response"] = self.memory_agent.get_initial_questions()
            print(f"[Router] check_profile - 档案不完整，请求用户填写")
        emit_trace("node_end", "check_profile", "执行完成")
        from agent.graph import trim_state
        trim_state(state)
        return state

    def classify_intent(self, state: AgentState) -> AgentState:
        """意图分类node"""
        print(f"[Router] classify_intent - 开始意图分类")
        emit_trace("node_start", "classify_intent", "正在识别用户意图...")
        state["route_decision"] = "classify_intent"
        intent_display_map = {
            "food": "食物分析",
            "food_report": "饮食记录",
            "workout": "运动指导",
            "workout_report": "运动记录",
            "recipe": "食谱推荐",
            "stats_query": "统计查询",
            "profile_update": "档案更新",
            "confirm": "确认操作",
            "general": "通用对话",
        }
        try:
            print(f"[Router] classify_intent - 用户输入: {state.get('input_message', '')[:50]}...")

            # 如果档案不完整，先处理档案创建/更新
            if not state.get("profile_complete", True):
                user_input = state.get("input_message", "")
                if any(c.isdigit() for c in user_input):
                    state["intent"] = "profile_update"
                else:
                    state["intent"] = "general"
                emit_event({"type": "intent", "intent": state["intent"]})
                emit_trace("classification", "classify_intent", f"识别为{intent_display_map.get(state['intent'], state['intent'])}")
                emit_trace("node_end", "classify_intent", "执行完成")
                return state

            image_info = state.get("image_info", {})
            if image_info.get("has_image", False):
                state["intent"] = "food"
                emit_event({"type": "intent", "intent": state["intent"]})
                emit_trace("classification", "classify_intent", f"识别为{intent_display_map.get(state['intent'], state['intent'])}")
                emit_trace("node_end", "classify_intent", "执行完成")
                return state

            # 检查是否是确认回答
            user_input = state.get("input_message", "").strip().lower()
            if self._is_confirmation(user_input):
                state["intent"] = "confirm"
                emit_event({"type": "intent", "intent": state["intent"]})
                emit_trace("classification", "classify_intent", f"识别为{intent_display_map.get(state['intent'], state['intent'])}")
                emit_trace("node_end", "classify_intent", "执行完成")
                return state

            # 检查是否是上下文相关的短回复（如"换一个"、"再来一个"等）
            context_intent = self._check_context_dependent_intent(state)
            if context_intent:
                state["intent"] = context_intent
                state["last_intent"] = context_intent
                emit_event({"type": "intent", "intent": state["intent"]})
                emit_trace("classification", "classify_intent", f"识别为{intent_display_map.get(state['intent'], state['intent'])}")
                emit_trace("node_end", "classify_intent", "执行完成")
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

            # 如果是主动报告，将列表中所有 food/workout 升级为 food_report/workout_report
            if is_reporting:
                valid_intents = [
                    "food_report" if i == "food" else ("workout_report" if i == "workout" else i)
                    for i in valid_intents
                ]
                # 如果升级后仍然是 general（即 LLM 没有检测到 food/workout），
                # 则直接根据关键词强制设置多意图
                if valid_intents == ["general"]:
                    detected = []
                    food_keywords = ["吃了", "吃了点", "吃了碗", "吃了份", "吃了些", "摄入了", "吃进去", "吃了一个", "吃了俩", "干掉", "干饭", "喝了"]
                    workout_keywords = ["跑了", "跑了步", "做了", "做了运动", "健身了", "练了", "锻炼了", "运动了", "跑步了", "游泳了", "骑车了", "走路了", "跳绳了", "打球了", "健身"]
                    text_lower = user_input.lower()
                    if any(kw in text_lower for kw in food_keywords):
                        detected.append("food_report")
                    if any(kw in text_lower for kw in workout_keywords):
                        detected.append("workout_report")
                    if detected:
                        valid_intents = detected
                        print(f"[Router] classify_intent - 关键词强制检测多意图: {valid_intents}")

            # 保留第一个作为主意图（兼容现有逻辑）
            state["intent"] = valid_intents[0]
            # 保存多意图列表
            state["intents"] = valid_intents
            # 保存上一个意图
            state["last_intent"] = valid_intents[0]
            print(f"[Router] classify_intent - 分类结果: {valid_intents}, last_intent: {valid_intents[0]}")
            emit_event({"type": "intent", "intent": state["intent"]})
            intent_labels = [intent_display_map.get(i, i) for i in valid_intents]
            emit_trace("classification", "classify_intent", f"识别为{' + '.join(intent_labels)}")
            emit_trace("node_end", "classify_intent", "执行完成")
        except Exception as e:
            print(f"[Router] classify_intent 错误: {e}")
            state["intent"] = "general"
            state["last_intent"] = None
            emit_event({"type": "intent", "intent": state["intent"]})
            emit_trace("classification", "classify_intent", "识别为通用对话")
            emit_trace("node_end", "classify_intent", "执行完成")
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
        """判断是否是确认回答（精确匹配）"""
        if not user_input:
            return False
        user_input_stripped = user_input.strip().lower()
        return user_input_stripped in _CONFIRM_WORDS_SET or user_input_stripped in _DENY_WORDS_SET

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
        emit_trace("node_start", "profile_node", "正在更新用户档案...")
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

        state["final_response"] = state["response"]
        emit_trace("node_end", "profile_node", "执行完成")
        return state

    def handle_confirm(self, state: AgentState) -> AgentState:
        """处理确认回答（兼容旧接口，优先使用 state 中的 pending_confirmation）

        新流程：graph 的 confirm_node → commit_node 负责实际的 commit 逻辑，
        这个方法主要用于从旧版 pending_stats.json 恢复兼容的情况。
        """
        print(f"[Router] handle_confirm - 处理确认")
        emit_trace("node_start", "confirm_recovery", "正在处理确认回复...")
        user_input = state.get("input_message", "").strip().lower()

        # 优先从 state 中的 pending_confirmation 读取
        pending_conf = state.get("pending_confirmation") or {}
        pending = self.memory_agent.load_pending_stats()

        if not pending_conf and not pending:
            state["response"] = "没有待确认的记录，请先查询食物或运动信息。"
            return state

        # 判断是肯定还是否定（精确匹配）
        is_deny = any(word == user_input for word in DENY_WORDS)
        is_yes = not is_deny and any(word == user_input for word in CONFIRM_WORDS)

        if is_yes:
            if pending.get("type") == "multi":
                if pending.get("food"):
                    self.memory_agent.update_daily_stats("meal", pending["food"])
                if pending.get("workout"):
                    self.memory_agent.update_daily_stats("workout", pending["workout"])
                state["response"] = "已计入统计（食物+运动）。\n\n" + self.memory_agent.get_daily_summary()
            else:
                self.memory_agent.update_daily_stats(pending["type"], pending["data"])
                summary = self.memory_agent.get_daily_summary()
                state["response"] = f"已计入统计。\n\n{summary}"
        else:
            state["response"] = f"好的，已取消。"

        self.memory_agent.clear_pending_stats()
        state["pending_stats"] = None
        emit_trace("node_end", "confirm_recovery", "执行完成")
        return state

    def handle_general(self, state: AgentState) -> AgentState:
        """一般对话node"""
        print(f"[Router] handle_general - 处理一般对话")
        emit_trace("node_start", "general_node", "正在思考回复...")
        try:
            if not state.get("profile_complete", True):
                state["response"] = self.memory_agent.get_initial_questions()
                state["final_response"] = state["response"]
                return state

            # 通过 ContextManager 统一构建消息列表
            ctx_mgr = get_context_manager()
            messages = ctx_mgr.build_prompt_messages("general", state)

            # 收集完整响应
            full_response = ""
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    full_response += chunk.content

            state["response"] = full_response
            state["final_response"] = full_response

            # 更新对话历史（使用 ContextManager 统一管理滑动窗口）
            ctx_mgr.append_messages(state, state["input_message"], full_response)
            print(f"[Router] handle_general - 更新后 messages 数量: {len(state.get('messages', []))}")
        except Exception as e:
            print(f"[Router] handle_general 错误: {e}")
            state["response"] = "抱歉，服务暂时不可用，请稍后重试。"
            state["final_response"] = state["response"]
        emit_trace("node_end", "general_node", "执行完成")
        return state

    def handle_stats_query(self, state: AgentState) -> AgentState:
        """处理统计查询"""
        print(f"[Router] handle_stats_query - 处理统计查询")
        summary = self.memory_agent.get_daily_summary()
        state["response"] = summary
        state["final_response"] = summary
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
