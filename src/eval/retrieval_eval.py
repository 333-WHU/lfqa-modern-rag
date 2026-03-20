from __future__ import annotations

import re
from typing import Any

from src.utils.logger import get_logger


LOGGER = get_logger(__name__)
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "was", "were",
    "be", "with", "as", "by", "at", "from", "that", "this", "it", "its", "their", "his", "her",
}


def tokenize(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS}


def lexical_overlap(a: str, b: str) -> float:
    a_tokens = tokenize(a)
    b_tokens = tokenize(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, len(a_tokens))


def extract_gold_targets(record: dict[str, Any], config: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    gold_cfg = config["gold_fields"]
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}

    gold_ids: list[str] = []
    gold_titles: list[str] = []
    gold_answers: list[str] = []

    for field in gold_cfg.get("passage_ids", []):
        value = metadata.get(field) or record.get(field)
        if isinstance(value, list):
            gold_ids.extend(str(item) for item in value)
        elif value:
            gold_ids.append(str(value))

    for field in gold_cfg.get("titles", []):
        value = metadata.get(field) or record.get(field)
        if isinstance(value, list):
            gold_titles.extend(str(item).strip().lower() for item in value)
        elif value:
            gold_titles.append(str(value).strip().lower())

    for field in gold_cfg.get("answers", []):
        value = metadata.get(field) or record.get(field)
        if isinstance(value, list):
            gold_answers.extend(str(item) for item in value if item)
        elif value:
            gold_answers.append(str(value))

    if record.get("answer"):
        gold_answers.append(str(record["answer"]))

    return gold_ids, gold_titles, gold_answers


def is_relevant(record: dict[str, Any], passage: dict[str, Any], config: dict[str, Any]) -> bool:
    gold_ids, gold_titles, gold_answers = extract_gold_targets(record, config)
    passage_id = str(passage.get("passage_id", ""))
    title = str(passage.get("title", "")).strip().lower()
    text = str(passage.get("text", ""))

    if gold_ids:
        return passage_id in set(gold_ids)
    if gold_titles:
        return title in set(gold_titles)
    threshold = float(config.get("answer_overlap_threshold", 0.2))
    return any(lexical_overlap(answer, text) >= threshold for answer in gold_answers if answer)


def evaluate_retrieval(records: list[dict[str, Any]], retrieval_cfg: dict[str, Any]) -> dict[str, Any]:
    ks = retrieval_cfg.get("ks", [1, 3, 5, 10])
    metrics: dict[str, Any] = {"sample_count": len(records), "ks": ks}
    if not records:
        return metrics

    for k in ks:
        hits = 0
        recall_sum = 0.0
        recall_count = 0
        for record in records:
            top_passages = record.get("retrieved_passages", [])[:k]
            relevant_flags = [is_relevant(record, passage, retrieval_cfg) for passage in top_passages]
            if any(relevant_flags):
                hits += 1

            gold_ids, gold_titles, gold_answers = extract_gold_targets(record, retrieval_cfg)
            gold_total = len(set(gold_ids)) or len(set(gold_titles))
            if gold_total > 0:
                matched = 0
                for passage in top_passages:
                    if is_relevant(record, passage, retrieval_cfg):
                        matched += 1
                recall_sum += matched / gold_total
                recall_count += 1
            else:
                recall_sum += 1.0 if any(relevant_flags) else 0.0
                recall_count += 1

        metrics[f"Hit@{k}"] = hits / len(records)
        metrics[f"Recall@{k}"] = recall_sum / max(1, recall_count)

    LOGGER.info("Computed retrieval metrics: %s", metrics)
    return metrics
