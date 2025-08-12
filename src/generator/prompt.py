BASE_PROMPT = """You are a helpful assistant. Answer strictly based on the provided context.
If the answer is not in the context, say you don't know.
Cite sources using [title#section] after each relevant sentence.

Question:
{question}

Context:
{context}
"""
def build_context(docs):
    parts = []
    for d in docs:
        md = d.get("metadata", {})
        parts.append(f"[{md.get('title','doc')}#{md.get('section',0)}] {d['text']}")
    return "\n\n".join(parts)

def make_prompt(question, docs):
    return BASE_PROMPT.format(question=question, context=build_context(docs))