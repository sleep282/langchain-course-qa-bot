"""Streamlit Web 界面：streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="课程文档问答助手", page_icon="📚", layout="wide")

st.title("📚 课程文档问答助手")
st.caption("基于 LangChain + RAG + 在线大模型 | 仅根据你上传的课程资料回答")

with st.sidebar:
    st.header("使用说明")
    st.markdown(
        """
1. 先把课程 PDF / 笔记放到 `data/` 目录
2. 命令行运行 `python build_index.py` 建立索引
3. 回到本页面直接提问

> 若修改了 `data/` 里的文档，请重新运行 `build_index.py`。
        """
    )
    if st.button("🔄 重新加载模型"):
        st.cache_resource.clear()
        st.success("已刷新")


@st.cache_resource(show_spinner="正在加载向量库与大模型...")
def load_chain():
    from src.rag_chain import build_rag_chain
    return build_rag_chain()


try:
    chain, retriever = load_chain()
except Exception as e:
    st.error(f"初始化失败：{e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是课程问答助手，可以根据你放入 data/ 的资料回答问题。"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("请输入你的问题，例如：什么是 RAG？"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索资料并思考..."):
            hits = retriever.invoke(question)
            with st.expander(f"📎 本次回答参考了 {len(hits)} 个资料片段"):
                for i, d in enumerate(hits):
                    src = d.metadata.get("source", "未知")
                    st.markdown(f"**片段 {i+1}** ← `{src}`")
                    st.write(d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""))
            answer = chain.invoke(question)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
