import os
import time
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings

from src.generator.llm import chat_complete

app = FastAPI(
    title="RAG Corporate API",
    description="Retrieval-Augmented Generation API for corporate documents",
    version="1.0.0"
)

# CORS middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "docs")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

class QuestionRequest(BaseModel):
    question: str

class Source(BaseModel):
    title: str
    page: int | None
    section: int
    source: str | None
    doc_id: str
    snippet: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Source]

def get_collection():
    """Get or create the ChromaDB collection."""
    try:
        return chroma_client.get_collection(name=CHROMA_COLLECTION)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Collection '{CHROMA_COLLECTION}' not found. Run build_index.py first.")

def retrieve_documents(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant documents from ChromaDB."""
    collection = get_collection()
    
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    if not results["documents"] or not results["documents"][0]:
        return []
    
    docs = []
    for i in range(len(results["documents"][0])):
        docs.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    
    return docs

def rerank_documents(docs: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
    """Simple reranking by distance (placeholder for cross-encoder)."""
    return sorted(docs, key=lambda x: x["distance"])

def build_prompt(question: str, docs: List[Dict[str, Any]]) -> str:
    """Build the prompt with context and question."""
    if not docs:
        return f"""Answer the following question based on your knowledge:

Question: {question}

Note: No relevant documents were found in the knowledge base."""

    context_parts = []
    for i, doc in enumerate(docs):
        metadata = doc["metadata"]
        title = metadata.get("title", "Unknown")
        page = metadata.get("page")
        section = metadata.get("section", 0)
        
        page_info = f"#p{page}" if page else ""
        citation = f"[{title}{page_info}-c{section}]"
        
        context_parts.append(f"{citation}\n{doc['text']}")
    
    context = "\n\n".join(context_parts)
    
    return f"""Answer the following question based ONLY on the provided context. If the answer cannot be found in the context, say "I cannot find this information in the provided documents."

Always cite your sources using the exact citation format provided in the context (e.g., [Document.pdf#p1-c0]).

Context:
{context}

Question: {question}

Answer:"""

def format_sources(docs: List[Dict[str, Any]]) -> List[Source]:
    """Format documents as sources for the response."""
    sources = []
    seen_combinations = set()
    
    for doc in docs:
        metadata = doc["metadata"]
        title = metadata.get("title", "Unknown")
        page = metadata.get("page")
        section = metadata.get("section", 0)
        source = metadata.get("source")
        doc_id = metadata.get("doc_id", "")
        
        # Create a unique key to avoid duplicates
        key = (title, page, section)
        if key in seen_combinations:
            continue
        seen_combinations.add(key)
        
        # Truncate snippet for display
        snippet = doc["text"][:300] + "..." if len(doc["text"]) > 300 else doc["text"]
        
        sources.append(Source(
            title=title,
            page=page,
            section=section,
            source=source,
            doc_id=doc_id,
            snippet=snippet
        ))
    
    return sources

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an answer with sources."""
    try:
        start_time = time.time()
        
        # Retrieve relevant documents
        docs = retrieve_documents(request.question, top_k=8)
        
        # Rerank documents
        ranked_docs = rerank_documents(docs, request.question)[:5]
        
        # Build prompt
        prompt = build_prompt(request.question, ranked_docs)
        
        # Generate answer
        answer = chat_complete(prompt, temperature=0.1, max_tokens=800)
        
        # Format sources
        sources = format_sources(ranked_docs)
        
        processing_time = time.time() - start_time
        print(f"[INFO] Question processed in {processing_time:.2f}s")
        
        return AnswerResponse(answer=answer, sources=sources)
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Corporate API",
        "version": "1.0.0",
        "endpoints": {
            "ask": "POST /ask - Ask a question",
            "health": "GET /health - Health check",
            "docs": "GET /docs - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)