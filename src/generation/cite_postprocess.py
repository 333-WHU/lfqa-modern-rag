from __future__ import annotations

import re
from typing import Any


CITATION_RE = re.compile(r"\[E(\d+)\]")


def extract_citations(text: str) -> list[str]:
    seen: list[str] = []
    for match in CITATION_RE.findall(text):
        tag = f"E{match}"
        if tag not in seen:
            seen.append(tag)
    return seen


def attach_citation_metadata(answer: str, evidences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations = extract_citations(answer)
    linked: list[dict[str, Any]] = []
    for citation in citations:
        try:
            index = int(citation[1:]) - 1
        except ValueError:
            continue
        if 0 <= index < len(evidences):
            linked.append(
                {
                    "citation": citation,
                    "passage_id": evidences[index].get("passage_id"),
                    "title": evidences[index].get("title"),
                }
            )
    return linked
