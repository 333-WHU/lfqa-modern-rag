from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.preprocess import load_processed_qa_split
from src.generation.generator import LocalGenerator
from src.generation.prompt_builder import build_prompt
from src.retrieval.bm25_search import BM25Searcher
from src.utils.io import ensure_dir, read_jsonl, write_jsonl
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def resolve_query_split(config: dict[str, Any]) -> tuple[str, str, Path]:
    dataset_name = config["retrieval"].get("default_dataset", "eli5")
    candidates = [config["retrieval"].get("default_split", "test"), "validation", "train", "all"]
    for split_name in candidates:
        try:
            split_path = load_processed_qa_split(config, dataset_name, split_name)
            return dataset_name, split_name, split_path
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No processed split found for dataset={dataset_name}. Run scripts/prepare_data.py first.")


class BaselinePipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.retrieval_cfg = config["retrieval"]
        self.generation_cfg = config["generation"]
        self.runtime_cfg = config["runtime"]
        self.paths_cfg = config["paths"]
        self.index_cfg = config["index"]
        self.generator = LocalGenerator(self.generation_cfg, self.runtime_cfg)

        mode = self.retrieval_cfg.get("baseline_mode", "bm25")
        if mode != "bm25":
            LOGGER.warning("Baseline mode %s is not supported in the new mainline. Falling back to bm25.", mode)
            mode = "bm25"
        self.searcher = BM25Searcher(self.index_cfg["bm25_dir"])
        self.mode = mode

    def run(self, max_samples: int | None = None) -> dict[str, Path]:
        dataset_name, split_name, split_path = resolve_query_split(self.config)
        records = read_jsonl(split_path)
        if max_samples is not None:
            records = records[:max_samples]

        output_root = ensure_dir(self.paths_cfg["output_dir"])
        retrieval_dir = ensure_dir(output_root / "retrieval_results")
        generation_dir = ensure_dir(output_root / "generations")
        run_name = f"baseline_{self.mode}_{dataset_name}_{split_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        generation_outputs: list[dict[str, Any]] = []
        retrieval_outputs: list[dict[str, Any]] = []
        top_k = int(self.retrieval_cfg.get("bm25_top_k", 8))
        max_evidences = int(self.generation_cfg.get("max_input_evidences", 5))

        for record in records:
            question = str(record["question"])
            retrieved = self.searcher.search(question, top_k=top_k)
            prompt = build_prompt(question, retrieved[:max_evidences], use_citation=False)
            generated = self.generator.generate(prompt, system_prompt=self.generation_cfg["system_prompt"])
            output_record = {
                "id": record["id"],
                "dataset": dataset_name,
                "split": split_name,
                "question": question,
                "answer": record.get("answer", ""),
                "retrieved_passages": retrieved,
                "final_answer": generated["text"],
                "citations": generated["citations"],
                "metadata": record.get("metadata", {}),
            }
            generation_outputs.append(output_record)
            retrieval_outputs.append(
                {
                    "id": record["id"],
                    "question": question,
                    "retrieved_passages": retrieved,
                    "mode": self.mode,
                }
            )

        retrieval_path = retrieval_dir / f"{run_name}.jsonl"
        generation_path = generation_dir / f"{run_name}.jsonl"
        write_jsonl(retrieval_outputs, retrieval_path)
        write_jsonl(generation_outputs, generation_path)
        LOGGER.info("Baseline pipeline completed. generation_path=%s", generation_path)
        return {
            "retrieval": retrieval_path,
            "generation": generation_path,
        }
