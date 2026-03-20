from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


class LocalReranker:
    def __init__(self, model_path, batch_size: int = 8, max_length: int = 512) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Local reranker model path not found: {model_path}")
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        LOGGER.info("Loading local reranker from %s on device=%s", model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        if not candidates:
            return []
        scored: list[dict[str, Any]] = []
        for start in range(0, len(candidates), self.batch_size):
            batch_candidates = candidates[start : start + self.batch_size]
            pairs = [(query, candidate.get("text", "")) for candidate in batch_candidates]
            batch = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.no_grad():
                logits = self.model(**batch).logits.squeeze(-1)
            logits_list = logits.detach().cpu().tolist()
            if isinstance(logits_list, float):
                logits_list = [logits_list]
            for candidate, score in zip(batch_candidates, logits_list):
                updated = dict(candidate)
                updated["rerank_score"] = float(score)
                scored.append(updated)
        ranked = sorted(scored, key=lambda item: float(item["rerank_score"]), reverse=True)[:top_k]
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
            item["retrieval_source"] = "reranked"
        return ranked
