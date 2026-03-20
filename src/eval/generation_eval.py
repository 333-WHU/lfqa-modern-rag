from __future__ import annotations

from typing import Any

from src.utils.logger import get_logger


LOGGER = get_logger(__name__)

try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover
    rouge_scorer = None


def evaluate_generation(records: list[dict[str, Any]], generation_cfg: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"sample_count": len(records)}
    if not records:
        return metrics

    if not generation_cfg.get("enable_rouge_l", True):
        metrics["rouge_l_enabled"] = False
        metrics["message"] = "ROUGE-L disabled in config."
        return metrics

    if rouge_scorer is None:
        metrics["rouge_l_enabled"] = False
        metrics["message"] = "rouge-score is not installed; skipped ROUGE-L."
        return metrics

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores: list[float] = []
    for record in records:
        reference = str(record.get("answer", "")).strip()
        prediction = str(record.get("final_answer", "")).strip()
        if not reference or not prediction:
            continue
        score = scorer.score(reference, prediction)["rougeL"].fmeasure
        scores.append(float(score))

    metrics["rouge_l_enabled"] = True
    metrics["ROUGE-L"] = sum(scores) / max(1, len(scores))
    metrics["evaluated_count"] = len(scores)
    LOGGER.info("Computed generation metrics: %s", metrics)
    return metrics
