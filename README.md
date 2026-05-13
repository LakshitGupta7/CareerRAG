# 🚀 CareerRAG – Resume + Job Description Analyzer

A **Retrieval-Augmented Generation (RAG)** application that analyzes resumes against job descriptions to generate **structured insights, match scores, skill gaps, and interview questions** — all backed by **evidence from the documents**.

---

## 📌 Features

- 📄 Upload Resume (PDF) + Job Description (PDF / pasted text)
- 🔍 **Hybrid retrieval** — Dense (FAISS) + Sparse (BM25) with Reciprocal Rank Fusion
- 🏆 **Cross-encoder re-ranking** for precision
- 🎯 Match Resume to JD with **score (0–100)**
- ⚠️ Identify **missing and weak skills**
- 🎤 Generate **targeted interview questions**
- 📊 Retrieval **confidence score** (sigmoid-normalized)
- ✅ **Validation layer** — second LLM pass checks if claims are grounded in context
- 🧾 Evidence-backed responses with **chunk-level citations** (`[resume-1]`, `[job_description-3]`)
- ⚡ **Real-time streaming** responses with robust model fallback
- 📥 **Export reports** as Markdown
- 🎨 Dark-mode Streamlit theme


## 🧠 Architecture

```
User Input → PDF Parsing (PyMuPDF)
    → Recursive Chunking (LangChain)
    → Dense Embeddings (Sentence Transformers) + BM25 Index
    → FAISS Vector Store
        ↓
Task-Specific Hybrid Retrieval (FAISS + BM25)
    → Reciprocal Rank Fusion (RRF)
    → Cross-Encoder Re-ranking
        ↓
Prompt Engineering (task-based templates)
    → LLM Streaming Generation (Groq — LLaMA)
        ↓
Validation Pass (grounding check)
    → Structured Output + UI Rendering
```


## ⚙️ Tech Stack

### 🧠 GenAI / ML
| Component | Technology |
|---|---|
| RAG framework | Custom pipeline |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Dense retrieval | FAISS (`IndexFlatIP`, cosine similarity) |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq API — `llama-3.1-8b-instant` + `llama-3.3-70b-versatile` fallback |
| Chunking | `RecursiveCharacterTextSplitter` (LangChain) |

### 📄 Processing
- **PyMuPDF** (`fitz`) for PDF text extraction
- Recursive character-level chunking with configurable overlap

### 🎨 Frontend
- **Streamlit** with custom dark theme
- Real-time streaming responses
- Markdown export

### 🛠️ DevOps & Infrastructure
| Component | Technology |
|---|---|
| IaC | Terraform (AWS) |
| CI/CD | Jenkins (Pipelines) |
| Container Registry | Docker Hub |
| Orchestration | Kubernetes (Self-managed Kubeadm) |
| Monitoring | Prometheus + Grafana |
| OS / Host | Ubuntu 22.04 LTS (AWS EC2) |

---

## 🛠️ DevOps Pipeline

The application is deployed using a complete end-to-end DevOps lifecycle, ensuring high availability, automated delivery, and deep observability.

### 1. Infrastructure as Code (Terraform)
- **AWS Provisioning**: Automated setup of 2x `c7i-flex.large` EC2 instances (Master & Worker).
- **Security Groups**: Dynamic configuration of security groups allowing ports for K8s (6443, 10250), NodePort (30000-32767), and Monitoring (9090, 3000).

### 2. CI/CD Pipeline (Jenkins)
- **Source Control**: GitHub integration triggers automated builds.
- **Docker Build**: Automated creation of lightweight, CPU-optimized Python images.
- **Push to Registry**: Secure push to Docker Hub (`lakshitgupta07/careerrag`).
- **Kubernetes Deployment**: Rolling updates to the cluster via `kubectl`.

### 3. Kubernetes Orchestration
- **Self-Managed Cluster**: Multi-node cluster built using `kubeadm`.
- **Deployment Strategy**: Replicas managed with rolling updates and resource limits (2Gi/3Gi RAM).
- **Service Layer**: NodePort service exposing the application on port `30851`.

### 4. Monitoring & Observability
- **Prometheus**: Native installation on the Master node scraping metrics from Node Exporters.
- **Node Exporter**: Deployed on both nodes to monitor CPU, RAM, Disk, and Network.
- **Grafana**: Interactive dashboards visualizing cluster health and application performance.

---


## 📂 Project Structure

```
CareerRAG/
├── app.py                  # Main Streamlit application
├── rag_pipeline.py         # Embeddings + FAISS + BM25 + RRF + cross-encoder
├── prompts.py              # Task-specific prompt templates
├── validator.py            # Grounding validation layer
├── ui_utils.py             # Output parsing + UI helpers
├── config.py               # Centralized configuration
├── requirements.txt        # Pinned dependencies
├── Dockerfile              # Container deployment
├── .dockerignore
├── .streamlit/
│   └── config.toml         # Streamlit theme
├── tests/
│   └── test_utils.py       # Unit tests
├── .env                    # API keys (not committed)
├── .gitignore
└── README.md
```


## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/LakshitGupta7/CareerRAG.git
cd CareerRAG
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # macOS / Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add API key
Create a `.env` file:
```
GROQ_API_KEY=your_api_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

### 🐳 Docker (alternative)
```bash
docker build -t careerrag .
docker run -p 8501:8501 --env-file .env careerrag
```


## 🧪 Run Tests
```bash
pip install pytest
pytest tests/ -v
```


## 🔍 How It Works

### 1. Document Processing
- Resume and JD are parsed using PyMuPDF
- Split into overlapping chunks using `RecursiveCharacterTextSplitter`

### 2. Index Building
- Each chunk is embedded with Sentence Transformers → stored in a FAISS index
- A parallel BM25 index is built for keyword matching

### 3. Hybrid Retrieval
- **Dense search** (FAISS) catches semantic similarity
- **Sparse search** (BM25) catches exact keyword matches
- **RRF** fuses both ranked lists into a single candidate pool
- **Cross-encoder** re-ranks the top candidates for precision

### 4. Generation
- Task-specific prompts enforce structured output with chunk citations
- Responses are streamed in real-time via Groq API
- Automatic fallback from `llama-3.1-8b` to `llama-3.3-70b` on failure

### 5. Validation
- A second LLM pass checks whether each claim is supported by the retrieved context
- Outputs: Supported / Partially Supported / Unsupported


## 📊 Output Examples

- **Candidate Overview** with cited evidence
- **Core Technical Skills** extracted from resume
- **Missing Skills** identified from JD comparison
- **Match Score** (0–100) with progress bar
- **Recommendations** for improvement
- **Evidence Used** (expandable chunk citations)
- **Validation Report** (Supported / Partial / Unsupported)


## 💡 Key Highlights

- **Hybrid retrieval** (semantic + keyword) with cross-encoder re-ranking
- **Evidence-backed** responses reduce hallucination
- **Validation layer** adds reliability
- **Streaming** for responsive UX
- **Source-aware chunk IDs** (`resume-1`, `job_description-3`) for accurate citations
- **Centralized config** — all constants in one file
- **Model caching** — ML models load once, persist across Streamlit reruns
- Clean separation of concerns


## 📜 License

MIT License

## 👨‍💻 Author

**Lakshit Gupta**
B.Tech CSE (AI/ML)
Lovely Professional University