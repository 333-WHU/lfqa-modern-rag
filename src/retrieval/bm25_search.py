from __future__ import annotations

from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.retrieval.bm25_index import tokenize_text
from src.utils.io import load_pickle, read_jsonl
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


class BM25Searcher:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        tokenized_path = self.index_dir / "tokenized_corpus.pkl"
        metadata_path = self.index_dir / "passages_metadata.jsonl"
        if not tokenized_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"BM25 index files missing in {self.index_dir}. Please run scripts/build_indexes.py first."
            )
        self.tokenized_corpus: list[list[str]] = load_pickle(tokenized_path)
        self.metadata: list[dict[str, Any]] = read_jsonl(metadata_path)
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        scores = self.bm25.get_scores(tokenize_text(query))
        ranked_indices = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)[:top_k]
        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            item = dict(self.metadata[idx])
            item.update(
                {
                    "score": float(scores[idx]),
                    "rank": rank,
                    "retrieval_source": "bm25",
                }
            )
            results.append(item)
        return results
