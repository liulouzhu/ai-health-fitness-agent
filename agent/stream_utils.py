"""流式事件工具函数（emit_trace / emit_event）

供 agent/graph.py 和 agent/router_agent.py 共同使用，避免循环导入。
"""
from langgraph.config import get_stream_writer


def emit_trace(stage: str, node: str = None, message: str = "", detail: dict = None):
    """向 stream custom 通道发送可视化进度事件"""
    try:
        writer = get_stream_writer()
        payload = {"type": "trace", "stage": stage, "message": message}
        if node is not None:
            payload["node"] = node
        if detail is not None:
            payload["detail"] = detail
        writer(payload)
    except Exception:
        pass


def emit_event(payload: dict):
    """发送自定义流事件（如 intent）"""
    try:
        writer = get_stream_writer()
        writer(payload)
    except Exception:
        pass
