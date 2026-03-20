from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


class BM25Searcher:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.db_path = self.index_dir / "bm25_fts.sqlite"
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"BM25 index database missing in {self.index_dir}. Please run scripts/build_indexes.py first."
            )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        connection = sqlite3.connect(str(self.db_path))
        try:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    passage_id,
                    article_id,
                    title,
                    section,
                    text,
                    bm25(passages, 2.0, 1.0, 1.0) AS score
                FROM passages
                WHERE passages MATCH ?
                ORDER BY score ASC
                LIMIT ?
                """,
                (query, top_k),
            ).fetchall()
        finally:
            connection.close()

        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            results.append(
                {
                    "passage_id": row["passage_id"],
                    "article_id": row["article_id"],
                    "title": row["title"],
                    "section": row["section"],
                    "text": row["text"],
                    "score": float(-row["score"]),
                    "rank": rank,
                    "retrieval_source": "bm25",
                }
            )
        return results
