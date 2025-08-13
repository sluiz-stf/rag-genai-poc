import os
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from langchain_openai import OpenAIEmbeddings
from src.generator.llm import chat_complete

load_dotenv()
app = FastAPI(title="RAG API")

# Embeddings para consultas (já foram usados na indexação)
embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))

# Cliente Chroma apontando para o mesmo path usado no build_index
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma")
CHROMA_COLL = os.getenv("CHROMA_COLLECTION", "docs")
_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _chroma_client.get_or_create_collection(CHROMA_COLL)

class Query(BaseModel):
    question: str

class SimpleChromaRetriever:
    def __init__(self, k: int = 8):
        self.k = k

    def query(self, question: str, embeddings_client: OpenAIEmbeddings) -> List[Dict[str, Any]]:
        # Geramos o embedding da query e passamos como query_embeddings,
        # porque a coleção foi populada com embeddings já calculados.
        qvec = embeddings_client.embed_query(question)
        res = _collection.query(
            query_embeddings=[qvec],
            n_results=self.k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res.get("documents", [[]])[0] if res.get("documents") else []
        metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
        dists = res.get("distances", [[]])[0] if res.get("distances") else []

        out = []
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({
                "document": doc,
                "metadata": meta or {},
                "score": float(dist) if dist is not None else None,  # menor é melhor (distância)
            })
        return out

class NoOpReranker:
    def rerank(self, question: str, raw_docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        # Ordena por distância crescente (se disponível) e corta em top_k
        scored = [d for d in raw_docs if d.get("score") is not None]
        others = [d for d in raw_docs if d.get("score") is None]
        scored.sort(key=lambda x: x["score"])
        reranked = scored + others
        return reranked[:top_k]

def make_prompt(question: str, docs: List[Dict[str, Any]]) -> str:
    # Concatena os melhores trechos como contexto
    context_parts = []
    for d in docs[:5]:
        meta = d.get("metadata", {}) or {}
        title = meta.get("title") or meta.get("source") or "doc"
        sec = meta.get("section")
        page = meta.get("page")
        tag = title
        if page is not None or sec is not None:
            extras = []
            if page is not None:
                extras.append(f"p{page}")
            if sec is not None:
                extras.append(f"c{sec}")
            tag = f"{title}#{'-'.join(extras)}"
        snippet = (d.get("document", "") or "").strip()
        if snippet:
            context_parts.append(f"[{tag}]\n{snippet}")
    context = "\n\n".join(context_parts)

    instructions = (
        "You are a helpful assistant. Use ONLY the context to answer. "
        "If the answer is not in the context, say you don't know."
    )
    return f"{instructions}\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer in the user's language."

retriever = SimpleChromaRetriever(k=8)
reranker = NoOpReranker()

def _cite(meta: Dict[str, Any]) -> str:
    title = (meta.get("title") or meta.get("source") or "doc")
    page = meta.get("page")
    sec = meta.get("section")
    parts = [title]
    tags = []
    if page is not None:
        tags.append(f"p{page}")
    if sec is not None:
        tags.append(f"c{sec}")
    return "[" + "#".join([parts[0], "-".join(tags)]) + "]" if tags else f"[{parts[0]}]"

@app.post("/ask")
def ask(q: Query):
    # 1) Recupera
    raw_docs = retriever.query(q.question, embeddings)

    # 2) Re-ranking
    top_docs = reranker.rerank(q.question, raw_docs, top_k=5)

    # 3) Prompt
    prompt_text = make_prompt(q.question, top_docs)

    # 4) Chat (formato messages)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context faithfully. If unsure, say you don't know."},
        {"role": "user", "content": prompt_text},
    ]
    answer_text = chat_complete(messages)

    # 5) Fontes deduplicadas com snippet
    seen = set()
    sources = []
    for d in top_docs:
        meta = d.get("metadata", {}) or {}
        doc_text = d.get("document", "") or ""
        key = (meta.get("doc_id"), meta.get("section"))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "title": meta.get("title") or meta.get("source"),
            "page": meta.get("page"),
            "section": meta.get("section"),
            "source": meta.get("source"),
            "doc_id": meta.get("doc_id"),
            "snippet": (doc_text[:300] + "...") if len(doc_text) > 300 else doc_text
        })

    # 6) Citações inline (3 primeiras)
    citations_inline = " ".join(_cite(s) for s in sources[:3]) if sources else ""
    final_answer = (answer_text + " " + citations_inline).strip()

    return {"answer": final_answer, "sources": sources}