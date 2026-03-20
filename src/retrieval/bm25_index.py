from __future__ import annotations

import sqlite3
from pathlib import Path

from src.utils.io import ensure_dir, iter_jsonl, write_json
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


class BM25Indexer:
    def __init__(self, output_dir: Path, batch_size: int = 5000) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.batch_size = batch_size

    def build(self, passages_path: Path) -> dict[str, Path]:
        if not passages_path.exists():
            raise FileNotFoundError(f"Passages file not found: {passages_path}")

        db_path = self.output_dir / "bm25_fts.sqlite"
        manifest_path = self.output_dir / "manifest.json"
        if db_path.exists():
            db_path.unlink()

        total_passages = count_jsonl_records(passages_path)
        if total_passages == 0:
            raise ValueError(f"No passages found in {passages_path}")

        connection = sqlite3.connect(str(db_path))
        try:
            connection.execute("PRAGMA journal_mode=WAL;")
            connection.execute("PRAGMA synchronous=NORMAL;")
            connection.execute("PRAGMA temp_store=MEMORY;")
            connection.execute("PRAGMA mmap_size=30000000000;")
            connection.execute(
                """
                CREATE VIRTUAL TABLE passages USING fts5(
                    passage_id UNINDEXED,
                    article_id UNINDEXED,
                    title,
                    section,
                    text,
                    tokenize='unicode61'
                );
                """
            )

            batch: list[tuple[str, str, str, str, str]] = []
            inserted = 0
            for record in iter_jsonl(passages_path):
                batch.append(
                    (
                        str(record.get("passage_id", "")),
                        str(record.get("article_id", "")),
                        str(record.get("title", "")),
                        str(record.get("section", "")),
                        str(record.get("text", "")),
                    )
                )
                if len(batch) >= self.batch_size:
                    connection.executemany(
                        "INSERT INTO passages (passage_id, article_id, title, section, text) VALUES (?, ?, ?, ?, ?)",
                        batch,
                    )
                    connection.commit()
                    inserted += len(batch)
                    LOGGER.info("BM25 FTS indexed %d/%d passages", inserted, total_passages)
                    batch.clear()

            if batch:
                connection.executemany(
                    "INSERT INTO passages (passage_id, article_id, title, section, text) VALUES (?, ?, ?, ?, ?)",
                    batch,
                )
                connection.commit()
                inserted += len(batch)

            write_json(
                {
                    "passages_path": str(passages_path),
                    "db_path": str(db_path),
                    "document_count": inserted,
                    "index_type": "sqlite_fts5_bm25",
                    "batch_size": self.batch_size,
                },
                manifest_path,
            )
            LOGGER.info("Built disk-backed BM25 FTS index at %s documents=%d", db_path, inserted)
            return {
                "db": db_path,
                "manifest": manifest_path,
            }
        finally:
            connection.close()
