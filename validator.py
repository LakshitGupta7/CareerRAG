def build_validation_prompt(answer_text, retrieved_chunks):
    context = "\n\n".join(
        [
            f"[{c['chunk_id']}] (source={c['source']}, score={c['score']:.4f})\n{c['text']}"
            for c in retrieved_chunks
        ]
    )

    return f"""
You are a strict answer-grounding validator.

Your job is to verify whether the answer is supported by the provided context.

Rules:
- Check whether the answer's major claims are supported by the retrieved context.
- If a claim is unsupported, list it clearly.
- If the answer is mostly supported, say so.
- Do NOT use outside knowledge.
- Be strict.
- Return output in exactly this format:

## Validation Status
Supported / Partially Supported / Unsupported

## Supported Claims
- ...

## Unsupported Claims
- ...

## Validator Notes
- ...

Context:
{context}

Answer to Validate:
{answer_text}
""".strip()