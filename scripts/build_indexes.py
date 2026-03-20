from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_index import BM25Indexer
from src.utils.config import load_project_config
from src.utils.logger import get_logger, setup_logging


LOGGER = get_logger(__name__)


def main() -> None:
    setup_logging()
    config = load_project_config(cwd=PROJECT_ROOT)
    passages_path = config["processing"]["wiki_passages_output"]
    if not passages_path.exists():
        raise FileNotFoundError(f"Passages file not found: {passages_path}. Please run scripts/prepare_data.py first.")

    bm25_indexer = BM25Indexer(config["index"]["bm25_dir"])
    bm25_outputs = bm25_indexer.build(passages_path)
    LOGGER.info("build_indexes finished. bm25=%s", bm25_outputs)
    LOGGER.info("Dense indexing is skipped in the new mainline. Full pipeline now uses BM25 first-stage plus reranker.")


if __name__ == "__main__":
    main()
