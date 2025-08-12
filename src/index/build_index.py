import os, json
import chromadb
from dotenv import load_dotenv
from src.ingest.parse_docs import load_raw_docs
from src.ingest.chunking import chunk_document
from langchain_openai import OpenAIEmbeddings

def main():
    load_dotenv()
    client = chromadb.PersistentClient(path="data/chroma")
    coll = client.get_or_create_collection("docs")
    embeds = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))

    docs = load_raw_docs()
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_document(d, max_tokens=400))

    # filtra chunks vazios
    cleaned = [c for c in all_chunks if c.get("text") and c["text"].strip()]
    if not cleaned:
        print("Nenhum texto para indexar. Verifique data/raw e o parser.")
        return

    ids = [c["chunk_id"] for c in cleaned]
    texts = [c["text"] for c in cleaned]
    metadatas = []
    for c in cleaned:
        metadatas.append({
            "doc_id": c.get("doc_id"),
            "title": c.get("title") or (c.get("source") and os.path.basename(c["source"])) or "document",
            "source": c.get("source") or c.get("title") or "unknown",
            "page": c.get("page"),              # pode ser None
            "section": c.get("section"),        # índice do chunk
        })

    # upsert
    vectors = embeds.embed_documents(texts)
    if not vectors:
        raise ValueError("Falha ao gerar embeddings (lista vazia).")

    coll.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors)
    print(f"Indexed {len(ids)} chunks")