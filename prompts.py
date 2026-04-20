"""
CareerRAG — Prompt Templates
Each task type has a structured prompt that enforces grounded, cited answers.
"""


def _build_context(retrieved_chunks: list[dict]) -> str:
    return "\n\n".join(
        [
            f"[{c['chunk_id']}] (source={c['source']}, score={c['score']:.4f})\n{c['text']}"
            for c in retrieved_chunks
        ]
    )


COMMON_RULES = """
Answer ONLY from the provided context.
Do NOT use outside knowledge.
Do NOT invent skills, projects, tools, companies, or achievements.
Every major claim must cite at least one chunk ID like [resume-1] or [job_description-2].
If information is missing, explicitly say so.
Be professional and concise.
Return valid markdown headings exactly as requested.
""".strip()


def build_prompt(task_type: str, user_question: str, retrieved_chunks: list[dict]) -> str:
    context = _build_context(retrieved_chunks)

    if task_type == "summarize":
        return f"""
You are a resume analysis assistant.

{COMMON_RULES}

Task:
Summarize the candidate's profile using only resume evidence.

Return exactly these sections:
## Candidate Overview
## Core Technical Skills
## Key Projects or Experience
## Strengths
## Evidence Used

Important:
- Use only resume evidence.
- Include chunk citations in the first four sections.

Context:
{context}
""".strip()

    elif task_type == "match":
        return f"""
You are a resume-job matching assistant.

{COMMON_RULES}

Task:
Analyze how well the resume matches the job description.

Return exactly these sections:
## Overall Match Summary
## Strong Matching Skills
## Missing or Weak Skills
## Match Score
## Recommendations
## Evidence Used

Important:
- Match Score must be a number from 0 to 100.
- Justify the score using cited evidence.
- Compare resume evidence against job description evidence.

Context:
{context}
""".strip()

    elif task_type == "missing_skills":
        return f"""
You are a skill-gap analysis assistant.

{COMMON_RULES}

Task:
Identify what the job description asks for that is not clearly supported by the resume.

Return exactly these sections:
## Missing Technical Skills
## Missing Experience Areas
## Weakly Supported Skills
## Top Priority Improvements
## Evidence Used

Important:
- Missing items must be supported by job description chunks.
- If a skill exists in the resume, do not mark it as missing.

Context:
{context}
""".strip()

    elif task_type == "interview_questions":
        return f"""
You are an interview preparation assistant.

{COMMON_RULES}

Task:
Generate targeted interview questions using the resume and job description.

Return exactly these sections:
## Technical Interview Questions
## Project-Based Questions
## Job-Description-Based Questions
## Behavioral Questions
## Why These Questions Were Chosen

Important:
- Do NOT return a generic summary.
- Each question group should be grounded in evidence.
- Mention chunk citations when explaining why questions were chosen.

Context:
{context}
""".strip()

    else:
        return f"""
You are a grounded document analysis assistant.

{COMMON_RULES}

Question:
{user_question}

Return a direct answer with chunk citations.

Context:
{context}
""".strip()