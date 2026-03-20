from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.full_pipeline import FullPipeline
from src.utils.config import load_project_config
from src.utils.logger import get_logger, setup_logging


LOGGER = get_logger(__name__)
MAX_SAMPLES: int | None = None


def main() -> None:
    setup_logging()
    config = load_project_config(cwd=PROJECT_ROOT)
    pipeline = FullPipeline(config)
    outputs = pipeline.run(max_samples=MAX_SAMPLES)
    LOGGER.info("run_full_pipeline finished. outputs=%s", outputs)


if __name__ == "__main__":
    main()
