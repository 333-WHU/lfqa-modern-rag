from __future__ import annotations

from typing import Any


def _format_evidence_block(evidences: list[dict[str, Any]], use_citation: bool) -> str:
    blocks: list[str] = []
    for idx, evidence in enumerate(evidences, start=1):
        label = f"[E{idx}]" if use_citation else f"Evidence {idx}"
        title = evidence.get("title", "")
        section = evidence.get("section", "")
        header = f"{label} Title: {title}"
        if section:
            header += f" | Section: {section}"
        blocks.append(f"{header}\n{evidence.get('text', '')}")
    return "\n\n".join(blocks)


def build_prompt(question: str, evidences: list[dict[str, Any]], use_citation: bool = False) -> str:
    evidence_block = _format_evidence_block(evidences, use_citation=use_citation)
    if use_citation:
        return (
            "Question:\n"
            f"{question}\n\n"
            "Evidence:\n"
            f"{evidence_block}\n\n"
            "Instruction:\n"
            "Write a detailed long-form answer. When you use evidence, cite it inline with markers like [E1] or [E2]. "
            "If the evidence is insufficient, say what remains uncertain."
        )
    return (
        "Question:\n"
        f"{question}\n\n"
        "Evidence:\n"
        f"{evidence_block}\n\n"
        "Instruction:\n"
        "Write a detailed long-form answer grounded in the provided evidence."
    )
