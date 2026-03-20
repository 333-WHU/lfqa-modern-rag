from __future__ import annotations

from typing import Any

from src.retrieval.bm25_search import BM25Searcher
from src.retrieval.dense_search import DenseSearcher
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def minmax_normalize(items: list[dict[str, Any]]) -> dict[str, float]:
    if not items:
        return {}
    scores = [float(item["score"]) for item in items]
    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score < 1e-12:
        return {str(item["passage_id"]): 1.0 for item in items}
    return {
        str(item["passage_id"]): (float(item["score"]) - min_score) / (max_score - min_score)
        for item in items
    }


class HybridSearcher:
    def __init__(self, bm25_searcher: BM25Searcher, dense_searcher: DenseSearcher, alpha: float = 0.5, beta: float = 0.5) -> None:
        self.bm25_searcher = bm25_searcher
        self.dense_searcher = dense_searcher
        self.alpha = alpha
        self.beta = beta

    def search(self, query: str, sparse_top_k: int = 10, dense_top_k: int = 10, final_top_k: int = 10) -> list[dict[str, Any]]:
        bm25_results = self.bm25_searcher.search(query, top_k=sparse_top_k)
        dense_results = self.dense_searcher.search(query, top_k=dense_top_k)

        sparse_scores = minmax_normalize(bm25_results)
        dense_scores = minmax_normalize(dense_results)

        combined: dict[str, dict[str, Any]] = {}
        for item in bm25_results + dense_results:
            passage_id = str(item["passage_id"])
            if passage_id not in combined:
                combined[passage_id] = dict(item)
            combined[passage_id]["sparse_score"] = sparse_scores.get(passage_id, 0.0)
            combined[passage_id]["dense_score"] = dense_scores.get(passage_id, 0.0)
            combined[passage_id]["score"] = self.alpha * combined[passage_id]["sparse_score"] + self.beta * combined[passage_id]["dense_score"]
            combined[passage_id]["retrieval_source"] = "hybrid"

        ranked = sorted(combined.values(), key=lambda item: float(item["score"]), reverse=True)[:final_top_k]
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        LOGGER.info("Hybrid retrieval produced %d results", len(ranked))
        return ranked
