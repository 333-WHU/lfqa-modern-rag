from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.utils.io import ensure_dir, write_jsonl
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def chunk_words(words: list[str], chunk_size: int, overlap: int) -> Iterable[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if not chunk:
            break
        yield chunk
        if start + chunk_size >= len(words):
            break


def build_passages_for_article(
    article: dict[str, str],
    chunk_size: int,
    overlap: int,
    min_words: int,
) -> list[dict[str, str]]:
    passages: list[dict[str, str]] = []
    text = article["text"]
    words = text.split()
    if not words:
        return passages

    article_id = article["id"]
    if len(words) <= chunk_size and len(words) >= min_words:
        return [
            {
                "passage_id": f"{article_id}_p0",
                "article_id": article_id,
                "title": article.get("title", ""),
                "section": article.get("section", ""),
                "text": text,
            }
        ]

    for idx, chunk in enumerate(chunk_words(words, chunk_size=chunk_size, overlap=overlap)):
        if len(chunk) < min_words:
            continue
        passages.append(
            {
                "passage_id": f"{article_id}_p{idx}",
                "article_id": article_id,
                "title": article.get("title", ""),
                "section": article.get("section", ""),
                "text": " ".join(chunk),
            }
        )
    return passages


def build_passages(
    wiki_records: list[dict[str, str]],
    chunk_size: int,
    overlap: int,
    min_words: int,
) -> list[dict[str, str]]:
    passages: list[dict[str, str]] = []
    for article in wiki_records:
        passages.extend(build_passages_for_article(article, chunk_size, overlap, min_words))
    LOGGER.info("Built wikipedia passages count=%d", len(passages))
    return passages


def save_passages(passages: list[dict[str, str]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    write_jsonl(passages, output_path)
    LOGGER.info("Saved passages to %s", output_path)
