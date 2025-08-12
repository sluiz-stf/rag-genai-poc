import chromadb
from langchain_openai import OpenAIEmbeddings

class VectorRetriever:
    def __init__(self, path="data/chroma", collection="docs", k=8):
        client = chromadb.PersistentClient(path=path)
        self.coll = client.get_or_create_collection(collection)
        self.k = k

    def query(self, q, embeddings: OpenAIEmbeddings):
        q_emb = embeddings.embed_query(q)
        res = self.coll.query(query_embeddings=[q_emb], n_results=self.k, include=["documents","metadatas","distances"])
        docs = []
        for i in range(len(res["ids"][0])):
            docs.append({
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "score": 1 - res["distances"][0][i]
            })
        return docs