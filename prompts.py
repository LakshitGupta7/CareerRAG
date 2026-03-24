# def _build_context(retrieved_chunks):
#     return "\n\n".join(
#         [f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks]
#     )


# def build_prompt(task_type, user_question, retrieved_chunks):
#     context = _build_context(retrieved_chunks)

#     common_rules = """
# Answer ONLY from the provided context.
# Do NOT use outside knowledge.
# Do NOT invent skills, projects, tools, or achievements.
# If information is missing, explicitly say so.
# Be precise and professional.
# """.strip()

#     if task_type == "summarize":
#         return f"""
# You are a resume analysis assistant.

# {common_rules}

# Task:
# Summarize the candidate's profile based only on the resume context.

# Output format:
# 1. Candidate Overview
# 2. Core Technical Skills
# 3. Key Projects or Experience
# 4. Strengths
# 5. Evidence Used

# Context:
# {context}
# """.strip()

#     elif task_type == "match":
#         return f"""
# You are a resume-job matching assistant.

# {common_rules}

# Task:
# Analyze how well the resume matches the job description.

# Output format:
# 1. Overall Match Summary
# 2. Strong Matching Skills
# 3. Missing or Weak Skills
# 4. Match Score (0-100)
# 5. Recommendations
# 6. Evidence Used

# Important:
# - Compare resume content against job description content.
# - The score must be justified from the context.
# - Do not give a high score unless strong alignment is clearly present.

# Context:
# {context}
# """.strip()

#     elif task_type == "missing_skills":
#         return f"""
# You are a skill-gap analysis assistant.

# {common_rules}

# Task:
# Identify skills, tools, technologies, or experience requested in the job description that are missing or weak in the resume.

# Output format:
# 1. Missing Technical Skills
# 2. Missing Experience Areas
# 3. Skills Present but Weakly Supported
# 4. Top 3 Priority Improvements
# 5. Evidence Used

# Important:
# - Only mention gaps that are supported by the job description and not clearly supported by the resume.
# - Do not repeat full summary sections.

# Context:
# {context}
# """.strip()

#     elif task_type == "interview_questions":
#         return f"""
# You are an interview preparation assistant.

# {common_rules}

# Task:
# Generate interview questions tailored to the candidate's resume and the job description.

# Output format:
# 1. Technical Interview Questions
# 2. Project-Based Questions
# 3. Job-Description-Based Questions
# 4. HR / Behavioral Questions
# 5. Why These Questions Were Chosen

# Important:
# - Do NOT give a general candidate summary.
# - Focus on question generation.
# - Questions must be grounded in the resume and/or job description.
# - Where useful, mention what part of the context triggered the question.

# Context:
# {context}
# """.strip()

#     else:
#         return f"""
# You are a document analysis assistant.

# {common_rules}

# Question:
# {user_question}

# Context:
# {context}
# """.strip()

def _build_context(retrieved_chunks):
    return "\n\n".join(
        [
            f"[{c['chunk_id']}] (source={c['source']}, score={c['score']:.4f})\n{c['text']}"
            for c in retrieved_chunks
        ]
    )


def build_prompt(task_type, user_question, retrieved_chunks):
    context = _build_context(retrieved_chunks)

    common_rules = """
Answer ONLY from the provided context.
Do NOT use outside knowledge.
Do NOT invent skills, projects, tools, companies, or achievements.
Every major claim must cite at least one chunk ID like [resume-1] or [job_description-2].
If information is missing, explicitly say so.
Be professional and concise.
Return valid markdown headings exactly as requested.
""".strip()

    if task_type == "summarize":
        return f"""
You are a resume analysis assistant.

{common_rules}

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

{common_rules}

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

{common_rules}

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

{common_rules}

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

{common_rules}

Question:
{user_question}

Return a direct answer with chunk citations.

Context:
{context}
""".strip()