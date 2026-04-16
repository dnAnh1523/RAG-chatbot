# 📚 Sổ Tay Sinh Viên - RAG Chatbot

A Vietnamese RAG (Retrieval-Augmented Generation) chatbot that helps students query information from the Student Handbook (Sổ Tay Sinh Viên) using AI.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-green.svg)](https://langchain.com)
[![Chainlit](https://img.shields.io/badge/Chainlit-2.9.6-orange.svg)](https://chainlit.io)

## ✨ Features

- 🔍 **Smart Retrieval** - Uses MMR (Maximal Marginal Relevance) search to find the most relevant information
- 🧠 **Context-Aware** - Maintains chat history for conversational follow-up questions
- 📄 **PDF Parsing** - Layout-aware extraction with table reconstruction for complex documents
- 🌐 **Vietnamese Support** - Optimized for Vietnamese text with bge-m3 multilingual embeddings
- ⚡ **GPU Accelerated** - Local embeddings run on CUDA (RTX 3050 compatible)
- 🐳 **Docker Ready** - Containerized deployment support

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   User      │────▶│   Chainlit   │────▶│  ChatGroq   │
│   Query     │     │     UI       │     │   (LLM)     │
└─────────────┘     └──────────────┘     └──────┬──────┘
                           │                     │
                           ▼                     │
                    ┌──────────────┐            │
                    │   ChromaDB   │◀───────────┘
                    │  (Vector DB) │    Retrieved
                    └──────┬───────┘    Context
                           │
                    ┌──────▼───────┐
                    │  bge-m3      │
                    │  Embeddings  │
                    │  (Local GPU) │
                    └──────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (optional but recommended)
- [Groq API Key](https://console.groq.com/keys)

### 1. Clone and Setup

```bash
# Clone the repository
cd sotay-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Create .env file
cp .env.example .env  # Or create manually

# Add your Groq API key to .env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Prepare Data

Place your PDF file at:
```
data/SO_TAY_SINH_VIEN.pdf
```

### 4. Build Vector Store

Run the ingestion pipeline once to parse the PDF and create the vector database:

```bash
python ingest.py
```

This will:
- Parse the PDF with layout-aware extraction
- Chunk documents semantically (800 tokens, 150 overlap)
- Generate embeddings using bge-m3 (~570MB model)
- Save to `./vectorstore/` directory

### 5. Run the Application

```bash
chainlit run app.py
```

Then open your browser at `http://localhost:8000`

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image (vectorstore must exist first!)
docker build -t sotay-rag .

# Run with environment variables
docker run -p 8000:8000 --env-file .env sotay-rag
```

**Note:** The vectorstore directory must be pre-built locally before building the Docker image.

## 📁 Project Structure

```
sotay-rag/
├── app.py              # Chainlit chatbot application
├── ingest.py           # PDF ingestion pipeline
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
├── chainlit.md         # UI welcome message
├── .env                # Environment variables (not in git)
├── .gitignore
├── data/               # PDF documents
│   └── SO_TAY_SINH_VIEN.pdf
└── vectorstore/        # ChromaDB vector database
    └── ...
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key for LLM inference | Yes |

### Model Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBED_MODEL` | `BAAI/bge-m3` | Multilingual embeddings model |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM for generation |
| `CHUNK_SIZE` | 800 | Document chunk size |
| `CHUNK_OVERLAP` | 150 | Chunk overlap for continuity |

## 💡 Usage Examples

The chatbot can answer questions about:

- 💰 **Học phí** - Tuition fees, payment deadlines, methods
- 📋 **Quy chế học vụ** - Academic regulations, exam rules
- 🎓 **Học bổng** - Scholarship policies and requirements
- 📅 **Lịch học** - Academic calendar, exam schedules
- 📜 **Quy định khác** - Other student handbook content

Example queries:
- "Học phí năm học 2024-2025 là bao nhiêu?"
- "Điều kiện xét học bổng khuyến khích học tập?"
- "Thời gian đăng ký học phần học kỳ 1?"

## 🔧 Development

### Tech Stack

- **LangChain 1.x** - LLM orchestration with LCEL (LangChain Expression Language)
- **Chainlit** - Chat UI framework
- **ChromaDB** - Vector database for document storage
- **Groq** - Fast LLM inference API
- **Unstructured** - PDF parsing with layout preservation
- **HuggingFace** - Local embeddings (bge-m3)

## Known Limitations

- Embedding runs on local GPU (CUDA required). Remove `"device": "cuda"` in both files to use CPU instead.
- Groq free tier has rate limits (~30 req/min) — fine for personal use.
- Complex tables spanning 3-4 pages are handled by `hi_res` strategy but may not be perfect.
- Chainlit is community-maintained as of May 2025 — stable but pin your version.

## 📝 License

This project is for educational purposes.

## 🙋 Support

For issues or questions, please check:
1. Ensure vectorstore is built (`python ingest.py`)
2. Verify GPU availability for embeddings
3. Check Groq API key is valid

---

**Made with ❤️ for Vietnamese students**
