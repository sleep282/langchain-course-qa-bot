"""向量库：ChromaDB 本地持久化。"""
from pathlib import Path

from langchain_chroma import Chroma

from .embeddings import get_embeddings

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"


def get_vector_store() -> Chroma:
    """加载已存在的向量库。"""
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            "向量库尚未构建，请先运行 `python build_index.py` 建立索引。"
        )
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


def build_vector_store(chunks) -> Chroma:
    """从文本片段新建并持久化向量库。"""
    CHROMA_DIR.mkdir(exist_ok=True)
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    print(f"  [入库] 向量库已保存到 {CHROMA_DIR}")
    return vs
