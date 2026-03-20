from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from src.eval.generation_eval import evaluate_generation
from src.eval.grounded_eval import evaluate_groundedness
from src.eval.retrieval_eval import evaluate_retrieval
from src.generation.cite_postprocess import attach_citation_metadata
from src.generation.generator import LocalGenerator
from src.generation.prompt_builder import build_prompt
from src.pipelines.baseline_pipeline import resolve_query_split
from src.retrieval.bm25_search import BM25Searcher
from src.retrieval.rerank import LocalReranker
from src.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


class FullPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.retrieval_cfg = config["retrieval"]
        self.generation_cfg = config["generation"]
        self.runtime_cfg = config["runtime"]
        self.eval_cfg = config["eval"]
        self.paths_cfg = config["paths"]
        self.model_cfg = config["models"]
        self.index_cfg = config["index"]

        self.first_stage_searcher = BM25Searcher(self.index_cfg["bm25_dir"])
        self.reranker = LocalReranker(
            model_path=self.model_cfg["bge_reranker_path"],
            batch_size=4,
            max_length=int(self.retrieval_cfg.get("reranker_max_length", self.retrieval_cfg.get("dense_max_length", 384))),
        )
        self.generator = LocalGenerator(self.generation_cfg, self.runtime_cfg)

    def run(self, max_samples: int | None = None) -> dict[str, Path]:
        dataset_name, split_name, split_path = resolve_query_split(self.config)
        records = read_jsonl(split_path)
        if max_samples is not None:
            records = records[:max_samples]

        output_root = ensure_dir(self.paths_cfg["output_dir"])
        retrieval_dir = ensure_dir(output_root / "retrieval_results")
        generation_dir = ensure_dir(output_root / "generations")
        metrics_dir = ensure_dir(output_root / "metrics")
        run_name = f"full_pipeline_{dataset_name}_{split_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        retrieval_outputs: list[dict[str, Any]] = []
        generation_outputs: list[dict[str, Any]] = []
        first_stage_top_k = int(self.retrieval_cfg.get("bm25_top_k", 20))
        rerank_top_k = int(self.retrieval_cfg.get("rerank_top_k", 5))
        max_evidences = int(self.generation_cfg.get("max_input_evidences", 5))

        for record in records:
            question = str(record["question"])
            bm25_candidates = self.first_stage_searcher.search(question, top_k=first_stage_top_k)
            reranked = self.reranker.rerank(question, bm25_candidates, top_k=rerank_top_k)
            prompt = build_prompt(question, reranked[:max_evidences], use_citation=True)
            generated = self.generator.generate(prompt, system_prompt=self.generation_cfg["citation_system_prompt"])
            output_record = {
                "id": record["id"],
                "dataset": dataset_name,
                "split": split_name,
                "question": question,
                "answer": record.get("answer", ""),
                "retrieved_passages": reranked,
                "first_stage_candidates": bm25_candidates,
                "final_answer": generated["text"],
                "citations": attach_citation_metadata(generated["text"], reranked),
                "metadata": record.get("metadata", {}),
            }
            retrieval_outputs.append(
                {
                    "id": record["id"],
                    "question": question,
                    "first_stage_candidates": bm25_candidates,
                    "retrieved_passages": reranked,
                    "mode": "bm25_rerank",
                }
            )
            generation_outputs.append(output_record)

        retrieval_path = retrieval_dir / f"{run_name}.jsonl"
        generation_path = generation_dir / f"{run_name}.jsonl"
        metrics_path = metrics_dir / f"{run_name}.json"
        write_jsonl(retrieval_outputs, retrieval_path)
        write_jsonl(generation_outputs, generation_path)

        metrics = {
            "retrieval": evaluate_retrieval(generation_outputs, self.eval_cfg["retrieval"]),
            "generation": evaluate_generation(generation_outputs, self.eval_cfg["generation"]),
            "groundedness": evaluate_groundedness(generation_outputs, self.eval_cfg["grounded"]),
        }
        write_json(metrics, metrics_path)
        LOGGER.info("Full pipeline completed. generation_path=%s metrics_path=%s", generation_path, metrics_path)
        return {
            "retrieval": retrieval_path,
            "generation": generation_path,
            "metrics": metrics_path,
        }
