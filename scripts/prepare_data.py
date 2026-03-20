from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import prepare_qa_datasets, prepare_wiki_passages
from src.utils.config import load_project_config
from src.utils.logger import get_logger, setup_logging


LOGGER = get_logger(__name__)


def main() -> None:
    setup_logging()
    config = load_project_config(cwd=PROJECT_ROOT)
    LOGGER.info("Preparing local QA datasets and wikipedia passages")

    qa_outputs = prepare_qa_datasets(config)
    passage_path = prepare_wiki_passages(config)
    LOGGER.info("prepare_data finished. qa_outputs=%s passage_path=%s", qa_outputs, passage_path)


if __name__ == "__main__":
    main()
