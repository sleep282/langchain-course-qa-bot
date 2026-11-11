"""文档加载与切分：支持 PDF / TXT / Markdown。"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_raw_documents(data_dir: Path = DATA_DIR):
    """递归读取 data/ 目录下所有 pdf / txt / md 文件。"""
    docs = []
    data_dir = Path(data_dir)

    for pdf in sorted(data_dir.rglob("*.pdf")):
        if pdf.name.startswith("."):
            continue
        print(f"  [加载] {pdf.relative_to(data_dir)}")
        docs.extend(PyPDFLoader(str(pdf)).load())

    for ext in ("*.txt", "*.md"):
        for f in sorted(data_dir.rglob(ext)):
            if f.name.startswith("."):
                continue
            print(f"  [加载] {f.relative_to(data_dir)}")
            try:
                docs.extend(TextLoader(str(f), encoding="utf-8").load())
            except UnicodeDecodeError:
                docs.extend(TextLoader(str(f), encoding="gbk").load())

    if not docs:
        raise RuntimeError(
            f"{data_dir} 目录下没有找到任何 .pdf / .txt / .md 文档，请先放入课程资料。"
        )
    return docs


def split_documents(raw_docs):
    """中文友好的递归切分，按段落/标点切分。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"  [切分] 共得到 {len(chunks)} 个文本片段")
    return chunks
