from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.retrieval.dense_index import LocalTextEmbedder
from src.utils.io import read_jsonl
from src.utils.logger import get_logger

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError("faiss is required for dense search. Install faiss-cpu locally.") from exc


LOGGER = get_logger(__name__)


class DenseSearcher:
    def __init__(
        self,
        model_path: Path,
        index_dir: Path,
        batch_size: int = 16,
        max_length: int = 512,
        query_instruction: str = "Represent this question for retrieving supporting documents:",
    ) -> None:
        index_path = index_dir / "passages.faiss"
        metadata_path = index_dir / "passages_metadata.jsonl"
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Dense index files missing in {index_dir}. Please run scripts/build_indexes.py first."
            )
        self.index = faiss.read_index(str(index_path))
        self.metadata: list[dict[str, Any]] = read_jsonl(metadata_path)
        self.embedder = LocalTextEmbedder(model_path=model_path, batch_size=batch_size, max_length=max_length)
        self.query_instruction = query_instruction

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        query_vector = self.embedder.encode([query], instruction=self.query_instruction)
        scores, indices = self.index.search(query_vector.astype(np.float32), top_k)
        results: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            item = dict(self.metadata[int(idx)])
            item.update(
                {
                    "score": float(score),
                    "rank": rank,
                    "retrieval_source": "dense",
                }
            )
            results.append(item)
        return results
