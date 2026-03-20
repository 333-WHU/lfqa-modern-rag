from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.utils.io import dump_pickle, ensure_dir, read_jsonl, write_json
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Indexer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = ensure_dir(output_dir)

    def build(self, passages_path: Path) -> dict[str, Path]:
        passages = read_jsonl(passages_path)
        if not passages:
            raise ValueError(f"No passages found in {passages_path}")

        tokenized_corpus = [tokenize_text(passage.get("text", "")) for passage in passages]
        _ = BM25Okapi(tokenized_corpus)

        tokenized_path = self.output_dir / "tokenized_corpus.pkl"
        metadata_path = self.output_dir / "passages_metadata.jsonl"
        manifest_path = self.output_dir / "manifest.json"

        dump_pickle(tokenized_corpus, tokenized_path)
        with metadata_path.open("w", encoding="utf-8") as f:
            for passage in passages:
                f.write(__import__("json").dumps(passage, ensure_ascii=False) + "\n")

        manifest = {
            "passages_path": str(passages_path),
            "tokenized_corpus_path": str(tokenized_path),
            "metadata_path": str(metadata_path),
            "document_count": len(passages),
        }
        write_json(manifest, manifest_path)
        LOGGER.info("Built BM25 resources at %s", self.output_dir)
        return {
            "tokenized_corpus": tokenized_path,
            "metadata": metadata_path,
            "manifest": manifest_path,
        }
