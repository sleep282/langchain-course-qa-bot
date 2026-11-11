# 📚 课程文档智能问答助手（基于 LangChain + RAG）

github链接：https://github.com/sleep282/langchain-course-qa-bot.git

---

## 一、功能特性

| 功能 | 说明 |
| --- | --- |
| 📄 **多格式文档接入** | 自动加载 `data/` 目录下的 PDF / TXT / Markdown |
| 🔍 **RAG 检索增强** | 先检索相关片段，再交给大模型作答，答案有据可查 |
| 🤖 **在线大模型可插拔** | 通过 OpenAI 兼容接口接入，一行配置切换 DeepSeek / 豆包 / OpenAI / Kimi / 通义 |
| 🧠 **本地中文 Embedding** | 使用开源 `BAAI/bge-small-zh-v1.5`，Embedding 阶段不花一分钱 API 费用 |
| 💾 **本地向量库** | ChromaDB 持久化到本地，重启不丢索引，无需额外启动数据库 |
| 💻 **双端交互** | 命令行 CLI + Streamlit 网页聊天界面 |
| 📎 **来源可追溯** | 每次回答都会展示命中了哪几个资料片段，方便老师验证"确实是基于文档答的" |

---

## 二、技术栈

- **框架**：LangChain（LCEL 表达式链）
- **大模型**：OpenAI 兼容 Chat Completions API（在线）
- **Embedding**：`sentence-transformers` + `BAAI/bge-small-zh-v1.5`（本地）
- **向量数据库**：ChromaDB
- **文档加载**：PyPDFLoader + TextLoader
- **文本切分**：RecursiveCharacterTextSplitter（中文标点友好）
- **界面**：Streamlit（Web）+ 原生 Python REPL（CLI）

---

## 三、项目结构

```
langchain-course-qa-bot/
├── README.md                  # 本说明文档
├── requirements.txt          # Python 依赖
├── .env.example              # 环境变量模板（复制为 .env 使用）
├── .gitignore
├── build_index.py            # ① 一键构建向量索引
├── cli.py                    # ② 命令行问答
├── app.py                    # ③ Streamlit Web 界面
├── src/
│   ├── __init__.py
│   ├── config.py             # 读取 .env 配置
│   ├── document_loader.py    # 加载 PDF/TXT/MD + 文本切分
│   ├── embeddings.py         # 本地 BGE 向量模型
│   ├── vector_store.py       # Chroma 向量库
│   └── rag_chain.py          # RAG 链（检索→Prompt→LLM）
├── data/
│   └── sample/
│       └── langchain_rag_notes.md   # 自带示例笔记，可直接跑通
└── chroma_db/                # 运行后自动生成，不入库
```

---

## 四、快速开始

### 1. 环境要求

- Python 3.9 ~ 3.11
- 一个在线大模型 API Key（推荐 **DeepSeek**，便宜且国内直连；也可用豆包/OpenAI）

### 2. 安装依赖

```bash
# 建议先建虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
cp .env.example .env        # Windows 用 copy .env.example .env
```

编辑 `.env`，填入你的 Key：

```ini
LLM_API_KEY=sk-你的key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

> 想换成豆包？把 `LLM_BASE_URL` 改成 `https://ark.cn-beijing.volces.com/api/v3`，`LLM_MODEL` 改成你的接入点 ID（`ep-xxx`）即可。

### 4. 放入课程资料

把课件 PDF、笔记 `.txt` / `.md` 丢进 `data/` 目录（子目录也行）。
项目自带了一份示例笔记在 `data/sample/`，不准备资料也能先跑通。

### 5. 构建索引（第一次必做）

```bash
python build_index.py
```

首次运行会自动下载 BGE 向量模型（约 100MB），之后离线可用。

### 6. 开始提问

**方式 A：命令行**

```bash
python cli.py
```

**方式 B：网页聊天界面（推荐演示用）**

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`，像聊天一样提问即可。

---

## 五、效果演示

```
你问> 什么是 RAG？

[检索到 3 个相关片段]
  · 片段1 <- langchain_rag_notes.md
  · 片段2 <- langchain_rag_notes.md
  · 片段3 <- langchain_rag_notes.md

助手> RAG 全称 Retrieval-Augmented Generation（检索增强生成）……
```

回答下方会展开"参考了哪几个资料片段"，老师可以一眼看到**答案确实来自你的文档**。

---

## 六、RAG 原理（作业答辩可讲）

```
                ┌─────────────── 离线阶段（build_index.py）──────────────┐
PDF/TXT/MD  ──►  Document Loader  ──►  Splitter  ──►  Embedding  ──►  Chroma
                                                                          │
                ┌─────────────── 在线阶段（cli.py / app.py）──────────────┘
用户问题  ──►  Embedding  ──►  Chroma 相似度检索 Top-K
                                          │
                                          ▼
                              拼 Prompt（资料片段 + 问题）
                                          │
                                          ▼
                              在线大模型（DeepSeek/豆包/...）──►  答案 + 来源
```

**关键设计点：**
1. **不把所有文档塞 Prompt**：上下文窗口有限，只检索 Top-K（默认 3）个最相关片段，省钱又精准；
2. **Prompt 里加了约束**：要求模型"只根据资料回答，资料里没有就明说不知道"，显著降低幻觉；
3. **Embedding 用本地模型**：调用大模型要花钱，Embedding 量大但不需要生成能力，用本地开源模型最划算；
4. **来源回溯**：每个 chunk 都保留了 `source` 元数据，回答时能展示出处。

---

## 七、常见问题

**Q：换了新文档要重新跑吗？**
A：要。把新文档放进 `data/` 后重新执行 `python build_index.py`。

**Q：可以接离线模型吗？**
A：可以。安装 [Ollama](https://ollama.com) 后 `ollama pull qwen2.5`，把 `.env` 里的 `LLM_BASE_URL` 改成 `http://localhost:11434/v1`，`LLM_MODEL` 改成 `qwen2.5` 即可，Embedding 本来就是本地的。

**Q：报错 `缺少环境变量 LLM_API_KEY`？**
A：没有复制 `.env.example` 为 `.env` 并填 Key。

**Q：第一次运行下载 Embedding 模型很慢？**
A：可在 HuggingFace 镜像站下载 `BAAI/bge-small-zh-v1.5` 后手动放到 `~/.cache/huggingface/`，或设置环境变量 `HF_ENDPOINT=https://hf-mirror.com`。

---

## 八、开源仓库

本项目已开源，欢迎 Star / Fork：

```
GitHub: https://github.com/sleep282/langchain-course-qa-bot.git
```

---

## 九、License

MIT
