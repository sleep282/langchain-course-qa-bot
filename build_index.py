"""一键构建索引：把 data/ 下的文档切分、向量化、写入 Chroma。"""
from src.document_loader import load_raw_documents, split_documents
from src.vector_store import build_vector_store


def main():
    print("== 第 1 步：加载课程文档 ==")
    raw = load_raw_documents()
    print(f"  共加载 {len(raw)} 个页面/段落")

    print("== 第 2 步：文本切分 ==")
    chunks = split_documents(raw)

    print("== 第 3 步：向量化入库 ==")
    build_vector_store(chunks)

    print("\n✅ 索引构建完成！接下来可以运行：")
    print("   - 命令行问答： python cli.py")
    print("   - Web 界面：   streamlit run app.py")


if __name__ == "__main__":
    main()
