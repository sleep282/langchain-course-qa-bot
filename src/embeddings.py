"""Embedding：使用本地开源中文向量模型 BGE，无需额外 API、离线可用。"""
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """首次运行会自动从 HuggingFace 下载模型（约 100MB），之后走本地缓存。"""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
