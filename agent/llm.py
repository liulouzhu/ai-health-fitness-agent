from dashscope import api_key
from langchain_openai import ChatOpenAI
from config import AgentConfig

def get_llm():
    llm = ChatOpenAI(
        model=AgentConfig.LLM_MODEL,
        api_key=AgentConfig.LLM_API_KEY,
        base_url=AgentConfig.LLM_BASE_URL,
    )
    return llm

def get_vlm():
    vlm = ChatOpenAI(
        model=AgentConfig.VLM_MODEL,
        api_key=AgentConfig.LLM_API_KEY,
        base_url=AgentConfig.LLM_BASE_URL,
    )
    return vlm