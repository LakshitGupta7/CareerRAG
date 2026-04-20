"""
CareerRAG — Main Streamlit Application
Resume + Job Description Analyzer powered by RAG.
"""

from __future__ import annotations

import os
import logging

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    PRIMARY_MODEL,
    FALLBACK_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    VALIDATION_TEMPERATURE,
    VALIDATION_MAX_TOKENS,
    EMBEDDING_MODEL,
    CROSS_ENCODER_MODEL,
)
from prompts import build_prompt
from validator import build_validation_prompt
from ui_utils import parse_sections, extract_match_score, get_validation_badge
from rag_pipeline import RAGPipeline, get_confidence_label

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="CareerRAG", layout="wide")
st.title("CareerRAG — Resume + JD Analyzer")
st.write("Analyze resumes against job descriptions using Retrieval-Augmented Generation (RAG).")

# ---------------------------
# Groq client
# ---------------------------
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()

client = Groq(api_key=groq_api_key)


# ---------------------------
# Cached model loading
# ---------------------------
@st.cache_resource(show_spinner="Loading ML models (first run only)...")
def load_models():
    """Load embedding + cross-encoder models once and cache across reruns."""
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    cross_enc = CrossEncoder(CROSS_ENCODER_MODEL)
    return embedder, cross_enc


embedder, cross_encoder = load_models()


# ---------------------------
# Session state
# ---------------------------
_DEFAULTS = {
    "rag": None,  # will be set below
    "documents_loaded": False,
    "selected_query": "",
    "selected_task_type": "summarize",
    "last_answer": "",
    "last_chunks": [],
    "chunk_count": 0,
    "last_confidence": "N/A",
    "last_validation": "",
    "last_validation_badge": "Unknown",
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.rag is None:
    st.session_state.rag = RAGPipeline(embedder=embedder, cross_encoder=cross_encoder)


# ---------------------------
# LLM helper — robust fallback
# ---------------------------
def _call_llm(model: str, user_prompt: str, system_prompt: str,
              temperature: float, max_tokens: int, stream: bool):
    """Single LLM call (returns response or generator)."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )
    if stream:
        return response  # raw stream object
    return response.choices[0].message.content


def call_groq(
    user_prompt: str,
    system_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stream: bool = False,
):
    """Call Groq with automatic model fallback.

    For non-streaming: returns the full text.
    For streaming: returns a *generator* that yields text chunks,
    with automatic fallback if the primary model fails before or during streaming.
    """
    if not stream:
        # Non-streaming — simple try/except
        try:
            return _call_llm(PRIMARY_MODEL, user_prompt, system_prompt,
                             temperature, max_tokens, stream=False)
        except Exception as e:
            logger.warning("Primary model (%s) failed: %s — falling back to %s",
                           PRIMARY_MODEL, e, FALLBACK_MODEL)
            return _call_llm(FALLBACK_MODEL, user_prompt, system_prompt,
                             temperature, max_tokens, stream=False)

    # Streaming — wrap in a generator so mid-stream failures trigger fallback
    def _safe_stream():
        try:
            raw_stream = _call_llm(PRIMARY_MODEL, user_prompt, system_prompt,
                                   temperature, max_tokens, stream=True)
            yielded_any = False
            for chunk in raw_stream:
                delta = chunk.choices[0].delta.content
                if delta is not None:
                    yielded_any = True
                    yield delta
            if yielded_any:
                return  # success
        except Exception as e:
            logger.warning("Primary stream failed: %s — retrying with %s",
                           e, FALLBACK_MODEL)

        # Fallback stream
        raw_stream = _call_llm(FALLBACK_MODEL, user_prompt, system_prompt,
                               temperature, max_tokens, stream=True)
        for chunk in raw_stream:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                yield delta

    return _safe_stream()


# ---------------------------
# Task-specific retrieval
# ---------------------------
def get_chunks_for_task(task_type: str, query: str, top_k: int) -> list[dict]:
    rag = st.session_state.rag

    if task_type == "summarize":
        return rag.retrieve(query, top_k=top_k, sources=["resume"])

    elif task_type == "match":
        half = max(2, top_k // 2 + 1)
        resume_chunks = rag.retrieve_from_source(query, "resume", top_k=half)
        jd_chunks = rag.retrieve_from_source(query, "job_description", top_k=half)
        combined = resume_chunks + jd_chunks
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]

    elif task_type == "missing_skills":
        jd_chunks = rag.retrieve_from_source(
            "required skills, tools, qualifications, responsibilities",
            "job_description", top_k=max(2, top_k),
        )
        resume_chunks = rag.retrieve_from_source(
            "candidate skills, tools, projects, experience",
            "resume", top_k=max(2, top_k),
        )
        combined = jd_chunks + resume_chunks
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]

    elif task_type == "interview_questions":
        resume_chunks = rag.retrieve_from_source(
            "projects, tools, technical skills, experience",
            "resume", top_k=max(2, top_k),
        )
        jd_chunks = rag.retrieve_from_source(
            "requirements, responsibilities, technologies",
            "job_description", top_k=max(2, top_k // 2 + 1),
        )
        combined = resume_chunks + jd_chunks
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]

    else:
        return rag.retrieve(query, top_k=top_k, sources=["resume", "job_description"])


# ---------------------------
# Sidebar — document upload
# ---------------------------
with st.sidebar:
    st.header("📄 Upload Documents")

    resume_file = st.file_uploader("Resume (PDF)", type=["pdf"])
    jd_file = st.file_uploader("Job Description PDF", type=["pdf"])
    jd_text = st.text_area("Or paste Job Description")

    top_k = st.slider("Context Chunks", 2, 8, 4)

    if st.button("Process Documents"):
        documents = []

        try:
            if resume_file:
                resume_text = st.session_state.rag.extract_text_from_pdf(resume_file)
                documents.append({"source": "resume", "text": resume_text})

            if jd_file:
                jd_content = st.session_state.rag.extract_text_from_pdf(jd_file)
                documents.append({"source": "job_description", "text": jd_content})
            elif jd_text.strip():
                documents.append({"source": "job_description", "text": jd_text.strip()})

            if not documents:
                st.warning("Upload at least one document.")
                st.session_state.documents_loaded = False
            else:
                with st.spinner("Building embeddings & indexes..."):
                    st.session_state.rag.build_vector_store(documents)

                st.session_state.documents_loaded = True
                st.session_state.chunk_count = st.session_state.rag.total_chunks()
                st.session_state.last_answer = ""
                st.session_state.last_chunks = []
                st.session_state.last_confidence = "N/A"
                st.session_state.last_validation = ""
                st.session_state.last_validation_badge = "Unknown"

                st.success("Documents processed successfully.")
                st.info(f"{st.session_state.chunk_count} total chunks created")

        except Exception as e:
            logger.exception("Document processing failed")
            st.session_state.documents_loaded = False
            st.error(f"Error: {e}")


# ---------------------------
# Status
# ---------------------------
st.subheader("Status")
if st.session_state.documents_loaded:
    st.success(f"Documents ready ({st.session_state.chunk_count} chunks)")
else:
    st.info("Upload and process documents to begin")


# ---------------------------
# Task selector
# ---------------------------
TASK_OPTIONS = {
    "Summarize Resume": {
        "task_type": "summarize",
        "query": "Summarize the candidate's resume.",
    },
    "Match Resume to JD": {
        "task_type": "match",
        "query": "Analyze how well the resume matches the job description.",
    },
    "Missing Skills": {
        "task_type": "missing_skills",
        "query": "Identify missing or weak skills compared to the job description.",
    },
    "Interview Questions": {
        "task_type": "interview_questions",
        "query": "Generate interview questions based on the resume and job description.",
    },
    "Custom Query": {
        "task_type": "generic",
        "query": "",
    },
}

st.subheader("Analysis Mode")

selected_mode = st.selectbox(
    "Choose analysis type",
    list(TASK_OPTIONS.keys()),
    index=0,
)

default_task_type = TASK_OPTIONS[selected_mode]["task_type"]
default_query = TASK_OPTIONS[selected_mode]["query"]

if selected_mode != "Custom Query":
    st.session_state.selected_task_type = default_task_type
    st.session_state.selected_query = default_query

query = st.text_input(
    "Enter your query",
    value=st.session_state.selected_query if selected_mode != "Custom Query" else "",
)


# ---------------------------
# Run analysis
# ---------------------------
if st.button("Run Analysis"):
    st.session_state.last_answer = ""
    st.session_state.last_chunks = []
    st.session_state.last_confidence = "N/A"
    st.session_state.last_validation = ""
    st.session_state.last_validation_badge = "Unknown"

    if not st.session_state.documents_loaded:
        st.warning("Process documents first.")
    elif not query.strip():
        st.warning("Enter a query.")
    else:
        task_type = default_task_type if selected_mode != "Custom Query" else "generic"

        with st.spinner("Retrieving relevant context..."):
            chunks = get_chunks_for_task(task_type, query, top_k)
            st.session_state.last_chunks = chunks
            st.session_state.last_confidence = get_confidence_label(chunks)

        if not chunks:
            st.warning("No relevant context found.")
        else:
            prompt = build_prompt(task_type, query, chunks)

            try:
                # Stream the answer
                stream_container = st.empty()
                with stream_container.container():
                    st.markdown("### Generating answer...")
                    answer_stream = call_groq(
                        user_prompt=prompt,
                        system_prompt="You are a precise AI assistant for grounded document analysis.",
                        temperature=DEFAULT_TEMPERATURE,
                        max_tokens=DEFAULT_MAX_TOKENS,
                        stream=True,
                    )
                    answer_text = st.write_stream(answer_stream)

                st.session_state.last_answer = answer_text
                stream_container.empty()

                # Validation pass (non-streaming)
                with st.spinner("Validating response..."):
                    validation_prompt = build_validation_prompt(answer_text, chunks)
                    validation_text = call_groq(
                        user_prompt=validation_prompt,
                        system_prompt="You are a strict validator for grounded document answers.",
                        temperature=VALIDATION_TEMPERATURE,
                        max_tokens=VALIDATION_MAX_TOKENS,
                        stream=False,
                    )

                st.session_state.last_validation = validation_text
                st.session_state.last_validation_badge = get_validation_badge(validation_text)

            except Exception as e:
                logger.exception("Generation/validation error")
                st.error(f"Generation/validation error: {e}")


# ---------------------------
# Output
# ---------------------------
if st.session_state.last_answer:
    st.markdown("## Analysis Result")

    answer_text = st.session_state.last_answer
    sections = parse_sections(answer_text)
    score = extract_match_score(answer_text)

    header_col1, header_col2 = st.columns([2, 1])

    with header_col1:
        if score is not None:
            st.metric("Match Score", f"{score}%")
            st.progress(score / 100)

    with header_col2:
        st.info(f"Retrieval Confidence: {st.session_state.last_confidence}")

    # Validation block
    st.markdown("### Validation")

    badge = st.session_state.last_validation_badge

    if badge == "Supported":
        st.success(f"Validation Status: {badge}")
    elif badge == "Partially Supported":
        st.warning(f"Validation Status: {badge}")
    elif badge == "Unsupported":
        st.error(f"Validation Status: {badge}")
    else:
        st.info(f"Validation Status: {badge}")

    if st.session_state.last_validation:
        with st.expander("View Detailed Validation Report", expanded=False):
            validation_sections = parse_sections(st.session_state.last_validation)

            if validation_sections:
                for title, content in validation_sections.items():
                    st.markdown(f"### {title}")
                    st.markdown(content)
                    st.divider()
            else:
                st.markdown(st.session_state.last_validation)

    # Render answer sections
    if sections:
        for title, content in sections.items():
            title_lower = title.lower()

            if "match score" in title_lower:
                st.markdown(f"### {title}")
                st.success(content)
            elif "missing" in title_lower:
                st.markdown(f"### {title}")
                st.warning(content)
            elif "recommend" in title_lower:
                st.markdown(f"### {title}")
                st.info(content)
            elif "evidence" in title_lower:
                with st.expander(title, expanded=False):
                    st.markdown(content)
            else:
                st.markdown(f"### {title}")
                st.markdown(content)

            st.divider()
    else:
        st.markdown(answer_text)

    # Export button
    st.download_button(
        label="📥 Download Report as Markdown",
        data=answer_text,
        file_name="careerrag_report.md",
        mime="text/markdown",
    )


# ---------------------------
# Retrieved Context
# ---------------------------
if st.session_state.last_chunks:
    with st.expander("View Retrieved Context"):
        for i, c in enumerate(st.session_state.last_chunks, 1):
            st.markdown(
                f"**{i}. {c['chunk_id']} | source={c['source']} | score={c['score']:.4f}**"
            )
            st.write(c["text"][:1000])