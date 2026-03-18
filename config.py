import os
from dotenv import load_dotenv
load_dotenv()

class AgentConfig:

    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL")
    VLM_MODEL = os.getenv("VLM_MODEL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")