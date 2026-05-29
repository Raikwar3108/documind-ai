# 📘 DocuMind AI – Hybrid RAG System

An advanced Retrieval-Augmented Generation (RAG) application built using **Streamlit**, **OpenAI Embeddings**, **Hybrid Retrieval (BM25 + Cosine Similarity)**, and **GPT-4o-mini**.

This project enables users to upload PDF documents and ask natural language questions while leveraging semantic search and hybrid retrieval techniques for accurate context-aware responses.

---

# 🚀 Features

* 📄 PDF Upload & Parsing
* ✂️ Recursive Character Text Chunking
* 🧠 OpenAI Embeddings (`text-embedding-3-small`)
* 🔍 Hybrid Retrieval

  * BM25 Retrieval
  * Cosine Similarity Search
* 🤖 GPT-4o-mini Answer Generation
* 📊 Retrieval Score Visualization
* 🎨 Modern Streamlit UI
* ⚡ Real-time Retrieval Pipeline

---

# 🧠 System Architecture
![Workflow Architecture](workflow.png)

# 🛠️ Tech Stack

| Component         | Technology                               |
| ----------------- | ---------------------------------------- |
| Frontend          | Streamlit                                |
| LLM               | GPT-4o-mini                              |
| Embeddings        | OpenAI text-embedding-3-small            |
| Retrieval         | BM25 + Cosine Similarity                 |
| PDF Parsing       | PyMuPDF (fitz)                           |
| Chunking          | LangChain RecursiveCharacterTextSplitter |
| Similarity Search | Scikit-learn                             |
| Hosting           | Streamlit Cloud                          |

---

# 📷 Screenshots

(Add screenshots here after deployment)

---

# ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/YOUR_USERNAME/documind-ai.git
```

Move into project folder:

```bash
cd documind-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run app.py
```

---

# 🔐 Environment Variables

Create `.streamlit/secrets.toml` or add Streamlit Cloud Secrets:

```toml
OPENAI_API_KEY="your_openai_api_key"
```

---

# 🌐 Deployment

This project is deployed using:

* Streamlit Cloud
* GitHub Integration

---

# 📈 Future Improvements

* Vector Database Integration (FAISS / Pinecone)
* Query Rewriting
* Reranking
* Multi-PDF Support
* Chat Memory
* Citation Highlighting
* Local Embedding Models
* Advanced RAG Evaluation

---

# 👨‍💻 Author

Prashant Raikwar

---

# ⭐ If you found this project useful

Please consider giving it a star on GitHub ⭐
