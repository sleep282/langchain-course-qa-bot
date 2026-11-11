"""RAG 链：检索 -> 拼 Prompt -> 大模型生成答案（LCEL 写法）。"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from .config import get_llm_config, get_retriever_k
from .vector_store import get_vector_store

SYSTEM_PROMPT = """你是一个严谨的课程学习助手。请严格根据下面提供的【检索到的课程资料片段】回答用户问题。

要求：
1. 只使用资料中的信息作答，不要编造、不要引入资料以外的知识；
2. 如果资料里没有相关内容，直接回答："根据提供的课程资料，暂时无法回答这个问题，建议补充相关文档。"
3. 回答条理清晰，必要时分点说明，并尽量引用资料中的关键概念。"""

USER_PROMPT = """【课程资料片段】
{context}

【用户问题】
{question}

【请回答】"""


def format_docs(docs) -> str:
    return "\n\n".join(
        f"—— 片段{i+1}（来源：{Path(d.metadata.get('source', '未知')).name}）——\n{d.page_content}"
        for i, d in enumerate(docs)
    )


# 顶部 import 补一个 Path（format_docs 用到）
from pathlib import Path  # noqa: E402


def build_llm() -> ChatOpenAI:
    cfg = get_llm_config()
    return ChatOpenAI(
        model=cfg["model"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        temperature=cfg["temperature"],
    )


def build_rag_chain():
    """返回 (chain, retriever)。chain.invoke(question) 直接得到答案字符串。"""
    vs = get_vector_store()
    retriever = vs.as_retriever(search_kwargs={"k": get_retriever_k()})

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("user", USER_PROMPT)]
    )
    llm = build_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever
