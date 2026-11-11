"""配置加载：统一从 .env 读取大模型参数。"""
import os
from dotenv import load_dotenv

load_dotenv()


def require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(
            f"缺少环境变量 {key}。请复制 .env.example 为 .env 并填入配置。"
        )
    return val


def get_llm_config() -> dict:
    return {
        "api_key": require_env("LLM_API_KEY"),
        "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        "model": os.getenv("LLM_MODEL", "deepseek-chat"),
        "temperature": float(os.getenv("TEMPERATURE", "0.3")),
    }


def get_retriever_k() -> int:
    return int(os.getenv("RETRIEVER_K", "3"))
