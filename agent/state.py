from typing import Annotated, Any, Dict, List, Optional, TypedDict
import operator


class AgentState(TypedDict):
    """State for the agent workflow"""

    # User context
    user_id: str
    session_id: Optional[str]

    # Input
    input_message: str
    image_data: Optional[str]

    # Processing
    intent: Optional[str]
    plan: Annotated[List[str], operator.add]
    current_agent: Optional[str]

    # Tool execution
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]
    tool_results: Annotated[List[Dict[str, Any]], operator.add]

    # Context from memory
    user_profile: Optional[Dict[str, Any]]
    daily_state: Optional[Dict[str, Any]]

    # Output
    response: Optional[str]
    actions_taken: Annotated[List[str], operator.add]

    # Control
    next_step: Optional[str]
    completed: bool
    error: Optional[str]