# import os
# import streamlit as st
# from dotenv import load_dotenv
# from groq import Groq
# from prompts import build_prompt
# from validator import build_validation_prompt
# from ui_utils import parse_sections, extract_match_score, get_validation_badge
# from rag_pipeline import RAGPipeline

# load_dotenv()

# st.set_page_config(page_title="CareerRAG", layout="wide")
# st.title("CareerRAG - Resume + JD Analyzer")
# st.write("Analyze resumes against job descriptions using Retrieval-Augmented Generation (RAG).")

# # ---------------------------
# # Groq setup
# # ---------------------------
# groq_api_key = os.getenv("GROQ_API_KEY")
# if not groq_api_key:
#     st.error("GROQ_API_KEY not found in .env file.")
#     st.stop()

# client = Groq(api_key=groq_api_key)

# # ---------------------------
# # Session state
# # ---------------------------
# if "rag" not in st.session_state:
#     st.session_state.rag = RAGPipeline()

# if "documents_loaded" not in st.session_state:
#     st.session_state.documents_loaded = False

# if "selected_query" not in st.session_state:
#     st.session_state.selected_query = ""

# if "selected_task_type" not in st.session_state:
#     st.session_state.selected_task_type = "summarize"

# if "last_answer" not in st.session_state:
#     st.session_state.last_answer = ""

# if "last_chunks" not in st.session_state:
#     st.session_state.last_chunks = []

# if "chunk_count" not in st.session_state:
#     st.session_state.chunk_count = 0

# if "last_confidence" not in st.session_state:
#     st.session_state.last_confidence = "N/A"

# if "last_validation" not in st.session_state:
#     st.session_state.last_validation = ""

# if "last_validation_badge" not in st.session_state:
#     st.session_state.last_validation_badge = "Unknown"

# # ---------------------------
# # Sidebar
# # ---------------------------
# with st.sidebar:
#     st.header("Upload Documents")

#     resume_file = st.file_uploader("Resume (PDF)", type=["pdf"])
#     jd_file = st.file_uploader("Job Description PDF", type=["pdf"])
#     jd_text = st.text_area("Or paste Job Description")

#     top_k = st.slider("Context Chunks", 2, 8, 4)

#     if st.button("Process Documents"):
#         documents = []

#         try:
#             if resume_file:
#                 resume_text = st.session_state.rag.extract_text_from_pdf(resume_file)
#                 documents.append({"source": "resume", "text": resume_text})

#             if jd_file:
#                 jd_content = st.session_state.rag.extract_text_from_pdf(jd_file)
#                 documents.append({"source": "job_description", "text": jd_content})
#             elif jd_text.strip():
#                 documents.append({"source": "job_description", "text": jd_text.strip()})

#             if not documents:
#                 st.warning("Upload at least one document.")
#                 st.session_state.documents_loaded = False
#             else:
#                 st.session_state.rag.build_vector_store(documents)
#                 st.session_state.documents_loaded = True
#                 st.session_state.chunk_count = st.session_state.rag.total_chunks()
#                 st.session_state.last_answer = ""
#                 st.session_state.last_chunks = []
#                 st.session_state.last_confidence = "N/A"

#                 st.success("Documents processed successfully.")
#                 st.info(f"{st.session_state.chunk_count} total chunks created")

#         except Exception as e:
#             st.session_state.documents_loaded = False
#             st.error(f"Error: {e}")

# # ---------------------------
# # Status
# # ---------------------------
# st.subheader("Status")
# if st.session_state.documents_loaded:
#     st.success(f"Documents ready ({st.session_state.chunk_count} chunks)")
# else:
#     st.info("Upload and process documents to begin")

# # ---------------------------
# # Task selector + quick actions
# # ---------------------------
# TASK_OPTIONS = {
#     "Summarize Resume": {
#         "task_type": "summarize",
#         "query": "Summarize the candidate's resume."
#     },
#     "Match Resume to JD": {
#         "task_type": "match",
#         "query": "Analyze how well the resume matches the job description."
#     },
#     "Missing Skills": {
#         "task_type": "missing_skills",
#         "query": "Identify missing or weak skills compared to the job description."
#     },
#     "Interview Questions": {
#         "task_type": "interview_questions",
#         "query": "Generate interview questions based on the resume and job description."
#     },
#     "Custom Query": {
#         "task_type": "generic",
#         "query": ""
#     }
# }

# st.subheader("Analysis Mode")

# selected_mode = st.selectbox(
#     "Choose analysis type",
#     list(TASK_OPTIONS.keys()),
#     index=list(TASK_OPTIONS.keys()).index("Summarize Resume")
# )

# default_task_type = TASK_OPTIONS[selected_mode]["task_type"]
# default_query = TASK_OPTIONS[selected_mode]["query"]

# if selected_mode != "Custom Query":
#     st.session_state.selected_task_type = default_task_type
#     st.session_state.selected_query = default_query

# query = st.text_input(
#     "Enter your query",
#     value=st.session_state.selected_query if selected_mode != "Custom Query" else ""
# )

# # ---------------------------
# # Confidence helper
# # ---------------------------
# def call_groq_with_fallback(user_prompt, system_prompt, temperature=0.2, max_tokens=800):
#     try:
#         response = client.chat.completions.create(
#             model="llama-3.1-8b-instant",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             temperature=temperature,
#             max_tokens=max_tokens
#         )
#         return response.choices[0].message.content

#     except Exception:
#         response = client.chat.completions.create(
#             model="llama-3.3-70b-versatile",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             temperature=temperature,
#             max_tokens=max_tokens
#         )
#         return response.choices[0].message.content

# def get_confidence_label(chunks):
#     if not chunks:
#         return "Low"

#     avg_score = sum(c["score"] for c in chunks) / len(chunks)

#     if avg_score >= 0.60:
#         return "High"
#     elif avg_score >= 0.35:
#         return "Medium"
#     else:
#         return "Low"

# # ---------------------------
# # Retrieval strategy
# # ---------------------------
# def get_chunks_for_task(task_type, query, top_k):
#     rag = st.session_state.rag

#     if task_type == "summarize":
#         return rag.retrieve(query, top_k=top_k, sources=["resume"])

#     elif task_type == "match":
#         resume_chunks = rag.retrieve_from_source(query, "resume", top_k=max(2, top_k // 2 + 1))
#         jd_chunks = rag.retrieve_from_source(query, "job_description", top_k=max(2, top_k // 2 + 1))
#         combined = resume_chunks + jd_chunks
#         combined.sort(key=lambda x: x["score"], reverse=True)
#         return combined[:top_k]

#     elif task_type == "missing_skills":
#         jd_chunks = rag.retrieve_from_source(
#             "required skills, tools, qualifications, responsibilities",
#             "job_description",
#             top_k=max(2, top_k)
#         )
#         resume_chunks = rag.retrieve_from_source(
#             "candidate skills, tools, projects, experience",
#             "resume",
#             top_k=max(2, top_k)
#         )
#         combined = jd_chunks + resume_chunks
#         combined.sort(key=lambda x: x["score"], reverse=True)
#         return combined[:top_k]

#     elif task_type == "interview_questions":
#         resume_chunks = rag.retrieve_from_source(
#             "projects, tools, technical skills, experience",
#             "resume",
#             top_k=max(2, top_k)
#         )
#         jd_chunks = rag.retrieve_from_source(
#             "requirements, responsibilities, technologies",
#             "job_description",
#             top_k=max(2, top_k // 2 + 1)
#         )
#         combined = resume_chunks + jd_chunks
#         combined.sort(key=lambda x: x["score"], reverse=True)
#         return combined[:top_k]

#     else:
#         return rag.retrieve(query, top_k=top_k, sources=["resume", "job_description"])

# # ---------------------------
# # Run analysis
# # ---------------------------
# if st.button("Run Analysis"):
#     st.session_state.last_answer = ""
#     st.session_state.last_chunks = []
#     st.session_state.last_confidence = "N/A"
#     st.session_state.last_validation = ""
#     st.session_state.last_validation_badge = "Unknown"

#     if not st.session_state.documents_loaded:
#         st.warning("Process documents first.")
#     elif not query.strip():
#         st.warning("Enter a query.")
#     else:
#         task_type = default_task_type if selected_mode != "Custom Query" else "generic"

#         with st.spinner("Retrieving relevant context..."):
#             chunks = get_chunks_for_task(task_type, query, top_k)
#             st.session_state.last_chunks = chunks
#             st.session_state.last_confidence = get_confidence_label(chunks)

#         if not chunks:
#             st.warning("No relevant context found.")
#         else:
#             prompt = build_prompt(task_type, query, chunks)

#             # with st.spinner("Generating response..."):
#             #     try:
#             #         response = client.chat.completions.create(
#             #             model="llama-3.1-8b-instant",
#             #             messages=[
#             #                 {
#             #                     "role": "system",
#             #                     "content": "You are a precise AI assistant for grounded document analysis."
#             #                 },
#             #                 {
#             #                     "role": "user",
#             #                     "content": prompt
#             #                 }
#             #             ],
#             #             temperature=0.2,
#             #             max_tokens=800
#             #         )
#             #         st.session_state.last_answer = response.choices[0].message.content

#             #     except Exception as e1:
#             #         try:
#             #             response = client.chat.completions.create(
#             #                 model="llama-3.3-70b-versatile",
#             #                 messages=[
#             #                     {
#             #                         "role": "system",
#             #                         "content": "You are a precise AI assistant for grounded document analysis."
#             #                     },
#             #                     {
#             #                         "role": "user",
#             #                         "content": prompt
#             #                     }
#             #                 ],
#             #                 temperature=0.2,
#             #                 max_tokens=800
#             #             )
#             #             st.session_state.last_answer = response.choices[0].message.content
#             #         except Exception as e2:
#             #             st.error(f"Groq error with primary model: {e1}")
#             #             st.error(f"Groq error with fallback model: {e2}")
#             with st.spinner("Generating response..."):
#                 try:
#                     answer_text = call_groq_with_fallback(
#                         user_prompt=prompt,
#                         system_prompt="You are a precise AI assistant for grounded document analysis.",
#                         temperature=0.2,
#                         max_tokens=800
#                     )

#                     st.session_state.last_answer = answer_text

#                     validation_prompt = build_validation_prompt(answer_text, chunks)

#                     validation_text = call_groq_with_fallback(
#                         user_prompt=validation_prompt,
#                         system_prompt="You are a strict validator for grounded document answers.",
#                         temperature=0.0,
#                         max_tokens=500
#                     )

#                     st.session_state.last_validation = validation_text
#                     st.session_state.last_validation_badge = get_validation_badge(validation_text)

#                 except Exception as e:
#                     st.error(f"Generation/validation error: {e}")

# # ---------------------------
# # Output
# # ---------------------------
# # ---------------------------
# # Output
# # ---------------------------
# # ---------------------------
# # Output
# # ---------------------------
# if st.session_state.last_answer:
#     st.markdown("## Analysis Result")

#     answer_text = st.session_state.last_answer
#     sections = parse_sections(answer_text)
#     score = extract_match_score(answer_text)

#     header_col1, header_col2 = st.columns([2, 1])

#     with header_col1:
#         if score is not None:
#             st.metric("Match Score", f"{score}%")
#             st.progress(score / 100)

#     with header_col2:
#         st.info(f"Retrieval Confidence: {st.session_state.last_confidence}")
#         st.markdown("### Validation")

#         badge = st.session_state.last_validation_badge

#         if badge == "Supported":
#             st.success(f"Validation Status: {badge}")
#         elif badge == "Partially Supported":
#             st.warning(f"Validation Status: {badge}")
#         elif badge == "Unsupported":
#             st.error(f"Validation Status: {badge}")
#         else:
#             st.info(f"Validation Status: {badge}")

#         if st.session_state.last_validation:
#             with st.expander("View Detailed Validation Report", expanded=False):
#                 validation_sections = parse_sections(st.session_state.last_validation)

#                 if validation_sections:
#                     for title, content in validation_sections.items():
#                         st.markdown(f"### {title}")
#                         st.markdown(content)
#                         st.divider()
#                 else:
#                     st.markdown(st.session_state.last_validation)

#     if sections:
#         for title, content in sections.items():
#             title_lower = title.lower()

#             if "match score" in title_lower:
#                 st.markdown(f"### {title}")
#                 st.success(content)

#             elif "missing" in title_lower:
#                 st.markdown(f"### {title}")
#                 st.warning(content)

#             elif "recommend" in title_lower:
#                 st.markdown(f"### {title}")
#                 st.info(content)

#             elif "evidence" in title_lower:
#                 with st.expander(title, expanded=False):
#                     st.markdown(content)

#             else:
#                 st.markdown(f"### {title}")
#                 st.markdown(content)

#             st.divider()
#     else:
#         st.markdown(answer_text)

# # ---------------------------
# # Retrieved Context
# # ---------------------------
# if st.session_state.last_chunks:
#     with st.expander("View Retrieved Context"):
#         for i, c in enumerate(st.session_state.last_chunks, 1):
#             st.markdown(
#                 f"**{i}. {c['chunk_id']} | source={c['source']} | score={c['score']:.4f}**"
#             )
#             st.write(c["text"][:1000])


import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from prompts import build_prompt
from validator import build_validation_prompt
from ui_utils import parse_sections, extract_match_score, get_validation_badge
from rag_pipeline import RAGPipeline

load_dotenv()

st.set_page_config(page_title="CareerRAG", layout="wide")
st.title("CareerRAG - Resume + JD Analyzer")
st.write("Analyze resumes against job descriptions using Retrieval-Augmented Generation (RAG).")

# ---------------------------
# Groq setup
# ---------------------------
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()

client = Groq(api_key=groq_api_key)

# ---------------------------
# Session state
# ---------------------------
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

if "selected_query" not in st.session_state:
    st.session_state.selected_query = ""

if "selected_task_type" not in st.session_state:
    st.session_state.selected_task_type = "summarize"

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = "N/A"

if "last_validation" not in st.session_state:
    st.session_state.last_validation = ""

if "last_validation_badge" not in st.session_state:
    st.session_state.last_validation_badge = "Unknown"


# ---------------------------
# Helper functions
# ---------------------------
def call_groq_with_fallback(user_prompt, system_prompt, temperature=0.2, max_tokens=800):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    except Exception:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


def get_confidence_label(chunks):
    if not chunks:
        return "Low"

    avg_score = sum(c["score"] for c in chunks) / len(chunks)

    if avg_score >= 0.60:
        return "High"
    elif avg_score >= 0.35:
        return "Medium"
    else:
        return "Low"


def get_chunks_for_task(task_type, query, top_k):
    rag = st.session_state.rag

    if task_type == "summarize":
        return rag.retrieve(query, top_k=top_k, sources=["resume"])

    elif task_type == "match":
        resume_chunks = rag.retrieve_from_source(query, "resume", top_k=max(2, top_k // 2 + 1))
        jd_chunks = rag.retrieve_from_source(query, "job_description", top_k=max(2, top_k // 2 + 1))
        combined = resume_chunks + jd_chunks
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]

    elif task_type == "missing_skills":
        jd_chunks = rag.retrieve_from_source(
            "required skills, tools, qualifications, responsibilities",
            "job_description",
            top_k=max(2, top_k)
        )
        resume_chunks = rag.retrieve_from_source(
            "candidate skills, tools, projects, experience",
            "resume",
            top_k=max(2, top_k)
        )
        combined = jd_chunks + resume_chunks
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]

    elif task_type == "interview_questions":
        resume_chunks = rag.retrieve_from_source(
            "projects, tools, technical skills, experience",
            "resume",
            top_k=max(2, top_k)
        )
        jd_chunks = rag.retrieve_from_source(
            "requirements, responsibilities, technologies",
            "job_description",
            top_k=max(2, top_k // 2 + 1)
        )
        combined = resume_chunks + jd_chunks
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]

    else:
        return rag.retrieve(query, top_k=top_k, sources=["resume", "job_description"])


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Upload Documents")

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
        "query": "Summarize the candidate's resume."
    },
    "Match Resume to JD": {
        "task_type": "match",
        "query": "Analyze how well the resume matches the job description."
    },
    "Missing Skills": {
        "task_type": "missing_skills",
        "query": "Identify missing or weak skills compared to the job description."
    },
    "Interview Questions": {
        "task_type": "interview_questions",
        "query": "Generate interview questions based on the resume and job description."
    },
    "Custom Query": {
        "task_type": "generic",
        "query": ""
    }
}

st.subheader("Analysis Mode")

selected_mode = st.selectbox(
    "Choose analysis type",
    list(TASK_OPTIONS.keys()),
    index=list(TASK_OPTIONS.keys()).index("Summarize Resume")
)

default_task_type = TASK_OPTIONS[selected_mode]["task_type"]
default_query = TASK_OPTIONS[selected_mode]["query"]

if selected_mode != "Custom Query":
    st.session_state.selected_task_type = default_task_type
    st.session_state.selected_query = default_query

query = st.text_input(
    "Enter your query",
    value=st.session_state.selected_query if selected_mode != "Custom Query" else ""
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

            with st.spinner("Generating response..."):
                try:
                    answer_text = call_groq_with_fallback(
                        user_prompt=prompt,
                        system_prompt="You are a precise AI assistant for grounded document analysis.",
                        temperature=0.2,
                        max_tokens=800
                    )

                    st.session_state.last_answer = answer_text

                    validation_prompt = build_validation_prompt(answer_text, chunks)

                    validation_text = call_groq_with_fallback(
                        user_prompt=validation_prompt,
                        system_prompt="You are a strict validator for grounded document answers.",
                        temperature=0.0,
                        max_tokens=500
                    )

                    st.session_state.last_validation = validation_text
                    st.session_state.last_validation_badge = get_validation_badge(validation_text)

                except Exception as e:
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

    # FULL-WIDTH VALIDATION BLOCK
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