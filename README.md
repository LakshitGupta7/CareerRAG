# 🚀 CareerRAG – Resume + Job Description Analyzer

A **Retrieval-Augmented Generation (RAG)** based GenAI application that analyzes resumes against job descriptions to generate **structured insights, match scores, skill gaps, and interview questions** — all backed by **evidence from the documents**.



## 📌 Features

- 📄 Upload Resume (PDF) + Job Description (PDF/Text)
- 🧠 Semantic + task-specific retrieval using embeddings
- 🔍 Match Resume to Job Description with **score (0–100)**
- ⚠️ Identify **missing and weak skills**
- 🎯 Generate **targeted interview questions**
- 📊 Retrieval **confidence score**
- ✅ **Validation layer** to check if answers are grounded in context
- 🧾 Evidence-backed responses with **chunk-level citations**
- 🎨 Clean UI with structured sections (Streamlit)


## 🧠 Architecture

User Input → Document Processing → Chunking → Embeddings → FAISS Vector DB
↓
Task-specific Retrieval (Resume / JD)
↓
Prompt Engineering (Task-based)
↓
LLM Generation (Groq - LLaMA)
↓
Validation Pass (Grounding Check)
↓
Structured Output + UI Rendering


## ⚙️ Tech Stack

### 🧠 GenAI / ML
- RAG (Retrieval-Augmented Generation)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS (Vector Database)
- Groq API (LLaMA Models)
  - `llama-3.1-8b-instant`
  - `llama-3.3-70b-versatile`

### 🔍 Retrieval
- Dense (semantic) retrieval  
- Task-specific retrieval strategies  
- Source-aware retrieval (resume vs JD)

### 📄 Processing
- PyMuPDF (`fitz`) for PDF parsing  
- Custom chunking (overlapping windows)

### 🧩 System Design
- `rag_pipeline.py` → embeddings + FAISS  
- `prompts.py` → task-specific prompts  
- `validator.py` → grounding validation  
- `ui_utils.py` → parsing + UI formatting  

### 🎨 Frontend
- Streamlit

### ⚙️ Backend
- Python  
- Virtual Environment (`venv`)  
- `python-dotenv`


## 📂 Project Structure
CareerRAG/
│
├── app.py # Main Streamlit app
├── rag_pipeline.py # Embeddings + FAISS + retrieval
├── prompts.py # Prompt engineering
├── validator.py # Validation layer
├── ui_utils.py # Output parsing + UI helpers
├── .env # API keys (not committed)
├── requirements.txt
└── README.md


## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/CareerRAG.git
cd CareerRAG
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add API key
```bash
GROQ_API_KEY=your_api_key_here
```

### 5. Run app
```bash
streamlit run app.py
```

## 🧪 Example Use Cases

- Resume screening for recruiters
- Candidate self-evaluation
- Interview preparation
- Skill gap analysis


## 🔍 How It Works

### 1. Document Processing
- Resume and JD are parsed using PyMuPDF
- Split into overlapping chunks

### 2. Embeddings & Storage
- Chunks are converted into embeddings
- Stored in FAISS vector database

### 3. Retrieval
Task-specific retrieval:
- Resume-only (summarization)
- Resume + JD (matching)
- JD-heavy (missing skills)

### 4. Generation
- LLM generates structured output using prompt engineering

### 5. Validation
- Second LLM pass checks if claims are supported by context


## 📊 Output Example

- Candidate Overview
- Core Technical Skills
- Missing Skills
- Match Score (with progress bar)
- Recommendations
- Evidence Used (chunk citations)
- Validation Report (Supported / Partial / Unsupported)


## 💡 Key Highlights

- Modular GenAI architecture
- Evidence-backed responses (reduces hallucination)
- Validation layer for reliability
- Clean separation of concerns
- Interview-ready system design


## 🔮 Future Improvements

- Hybrid retrieval (semantic + keyword)
- PDF report export
- Highlight unsupported claims in output
- Multi-model comparison
- Deployment (Docker / cloud)


## 📜 License

MIT License

## 👨‍💻 Author

**Lakshit Gupta**  
B.Tech CSE (AI/ML)  
Lovely Professional University