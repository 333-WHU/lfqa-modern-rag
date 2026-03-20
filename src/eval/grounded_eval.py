from __future__ import annotations

import re
from typing import Any

from src.eval.retrieval_eval import lexical_overlap
from src.generation.cite_postprocess import extract_citations
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def evaluate_groundedness(records: list[dict[str, Any]], grounded_cfg: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"sample_count": len(records)}
    if not records:
        return metrics

    threshold = float(grounded_cfg.get("lexical_overlap_threshold", 0.15))
    citation_required = bool(grounded_cfg.get("require_citation", True))

    citation_presence = 0
    valid_citation_count = 0
    support_scores: list[float] = []

    for record in records:
        answer = str(record.get("final_answer", ""))
        evidences = record.get("retrieved_passages", [])
        citations = extract_citations(answer)
        if citations:
            citation_presence += 1

        valid = True
        for citation in citations:
            try:
                idx = int(citation[1:]) - 1
            except ValueError:
                valid = False
                continue
            if idx < 0 or idx >= len(evidences):
                valid = False
        if valid:
            valid_citation_count += 1

        sentences = [segment.strip() for segment in SENTENCE_RE.split(answer) if segment.strip()]
        sentence_support: list[float] = []
        for sentence in sentences:
            cited_ids = extract_citations(sentence)
            candidate_texts: list[str] = []
            if cited_ids:
                for citation in cited_ids:
                    idx = int(citation[1:]) - 1
                    if 0 <= idx < len(evidences):
                        candidate_texts.append(str(evidences[idx].get("text", "")))
            else:
                candidate_texts = [str(item.get("text", "")) for item in evidences]

            if not candidate_texts:
                sentence_support.append(0.0)
                continue
            best_overlap = max(lexical_overlap(sentence, evidence_text) for evidence_text in candidate_texts)
            sentence_support.append(best_overlap)

        support_scores.append(sum(score >= threshold for score in sentence_support) / max(1, len(sentence_support)))

    metrics["citation_presence_rate"] = citation_presence / len(records)
    metrics["valid_citation_rate"] = valid_citation_count / len(records)
    metrics["grounded_sentence_rate"] = sum(support_scores) / max(1, len(support_scores))
    metrics["citation_required"] = citation_required
    LOGGER.info("Computed groundedness metrics: %s", metrics)
    return metrics
