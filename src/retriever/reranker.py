from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, docs, top_k=5):
        pairs = [(query, d["text"]) for d in docs]
        scores = self.model.predict(pairs).tolist()
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return docs[:top_k]