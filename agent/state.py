from typing import Annotated, Any, Dict, List, Optional, TypedDict
import operator


class AgentState(TypedDict):
    input_message: str
    intent: str