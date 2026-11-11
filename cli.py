"""命令行交互问答。"""
from src.rag_chain import build_rag_chain


def main():
    print("正在加载模型与向量库...")
    chain, retriever = build_rag_chain()
    print("=" * 60)
    print("课程文档问答助手已就绪！输入问题开始提问，输入 exit 退出。")
    print("=" * 60)

    while True:
        try:
            question = input("\n你问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("再见！")
            break

        # 先打印命中了哪些资料片段（便于老师看到检索过程）
        hits = retriever.invoke(question)
        print(f"\n[检索到 {len(hits)} 个相关片段]")
        for i, d in enumerate(hits):
            src = d.metadata.get("source", "未知")
            print(f"  · 片段{i+1} <- {src}")

        answer = chain.invoke(question)
        print(f"\n助手> {answer}")


if __name__ == "__main__":
    main()
