from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.data.loaders import LocalDatasetLoader, SUPPORTED_SUFFIXES, pick_first
from src.data.build_wiki_passages import build_passages_for_article
from src.utils.io import ensure_dir, iter_records_from_file, write_jsonl
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, lowercase: bool = False, normalize_whitespace: bool = True) -> str:
    processed = text.strip()
    if normalize_whitespace:
        processed = WHITESPACE_RE.sub(" ", processed)
    if lowercase:
        processed = processed.lower()
    return processed


def _stringify_answer(answer_value: Any) -> str:
    if answer_value is None:
        return ""
    if hasattr(answer_value, "tolist") and not isinstance(answer_value, (str, bytes, bytearray)):
        try:
            answer_value = answer_value.tolist()
        except Exception:
            pass
    if isinstance(answer_value, str):
        return answer_value
    if isinstance(answer_value, (list, tuple, set)):
        segments = [_stringify_answer(item) for item in answer_value]
        return "\n".join(segment for segment in segments if segment)
    if isinstance(answer_value, dict):
        for key in ("answer", "text", "long_answer", "response", "body", "short_answers"):
            if key in answer_value and answer_value[key] not in (None, "", []):
                return _stringify_answer(answer_value[key])
        return json.dumps(answer_value, ensure_ascii=False)
    return str(answer_value)


def _build_eli5_question(record: dict[str, Any]) -> str | None:
    title = str(record.get("title", "")).strip()
    selftext = str(record.get("selftext", "")).strip()
    if title and selftext:
        return f"{title}\n\n{selftext}"
    if title:
        return title
    if selftext:
        return selftext
    return None


def normalize_qa_record(
    record: dict[str, Any],
    dataset_name: str,
    field_map: dict[str, list[str]],
    lowercase: bool = False,
    normalize_whitespace_flag: bool = True,
) -> dict[str, Any] | None:
    if dataset_name == "eli5":
        question = _build_eli5_question(record)
        answer_value = pick_first(record, field_map.get("answer", ["answers.text", "answer"]))
    else:
        question = pick_first(record, field_map.get("question", ["question"]))
        answer_value = pick_first(record, field_map.get("answer", ["answer"]))

    if question is None:
        return None

    answer = _stringify_answer(answer_value)
    sample_id = pick_first(record, field_map.get("id", ["id"])) or f"{dataset_name}_{abs(hash(str(question)))}"

    normalized = {
        "id": str(sample_id),
        "dataset": dataset_name,
        "question": normalize_text(str(question), lowercase, normalize_whitespace_flag),
        "answer": normalize_text(answer, lowercase, normalize_whitespace_flag) if answer else "",
        "metadata": record,
    }
    return normalized


def normalize_wiki_record(
    record: dict[str, Any],
    field_map: dict[str, list[str]],
    lowercase: bool = False,
    normalize_whitespace_flag: bool = True,
) -> dict[str, Any] | None:
    text = pick_first(record, field_map.get("text", ["text"]))
    if not text:
        return None
    title = pick_first(record, field_map.get("title", ["title"])) or "Untitled"
    section = pick_first(record, field_map.get("section", ["section"])) or ""
    article_id = pick_first(record, field_map.get("id", ["id"])) or f"wiki_{abs(hash(str(title) + str(section) + str(text)))}"
    return {
        "id": str(article_id),
        "title": normalize_text(str(title), lowercase, normalize_whitespace_flag),
        "section": normalize_text(str(section), lowercase, normalize_whitespace_flag),
        "text": normalize_text(str(text), lowercase, normalize_whitespace_flag),
        "metadata": record,
    }


def prepare_qa_datasets(config: dict[str, Any]) -> dict[str, dict[str, Path]]:
    dataset_cfg = config["datasets"]
    path_cfg = config["paths"]
    processing_cfg = config["processing"]
    output_dir = ensure_dir(processing_cfg["qa_output_dir"])
    outputs: dict[str, dict[str, Path]] = {}

    for dataset_name in ("eli5", "asqa"):
        current_cfg = dataset_cfg.get(dataset_name, {})
        if not current_cfg.get("enabled", False):
            continue
        path_key = current_cfg["path_key"]
        loader = LocalDatasetLoader(dataset_name, current_cfg, path_cfg[path_key])
        loaded = loader.load()
        field_map = current_cfg.get("field_map", {})
        outputs[dataset_name] = {}
        for split_name, records in loaded.items():
            normalized_records = []
            for record in records:
                normalized = normalize_qa_record(
                    record,
                    dataset_name=dataset_name,
                    field_map=field_map,
                    lowercase=processing_cfg.get("lowercase_text", False),
                    normalize_whitespace_flag=processing_cfg.get("normalize_whitespace", True),
                )
                if normalized is None:
                    continue
                if normalized["answer"] or processing_cfg.get("include_empty_answer", False):
                    normalized_records.append(normalized)
            split_file = output_dir / f"{dataset_name}_{split_name}.jsonl"
            write_jsonl(normalized_records, split_file)
            outputs[dataset_name][split_name] = split_file
            LOGGER.info("Wrote normalized QA dataset %s split=%s count=%d -> %s", dataset_name, split_name, len(normalized_records), split_file)

    return outputs


def prepare_wiki_passages(config: dict[str, Any]) -> Path:
    dataset_cfg = config["datasets"]["wikipedia"]
    path_cfg = config["paths"]
    processing_cfg = config["processing"]
    wiki_path = path_cfg[dataset_cfg["path_key"]]
    output_path = processing_cfg["wiki_passages_output"]
    ensure_dir(output_path.parent)
    field_map = dataset_cfg.get("field_map", {})

    if wiki_path.is_file():
        files = [wiki_path]
    else:
        files = sorted(
            file for file in wiki_path.rglob("*") if file.is_file() and file.suffix.lower() in SUPPORTED_SUFFIXES
        )
    if not files:
        raise FileNotFoundError(f"No supported wikipedia files found under {wiki_path}")

    passage_count = 0
    for file_index, file_path in enumerate(files, start=1):
        LOGGER.info("Processing wikipedia source file %d/%d: %s", file_index, len(files), file_path)
        file_passage_count = 0
        with output_path.open("a" if output_path.exists() and passage_count > 0 else "w", encoding="utf-8") as writer:
            for record in iter_records_from_file(file_path, parquet_batch_size=2000):
                normalized = normalize_wiki_record(
                    record,
                    field_map=field_map,
                    lowercase=processing_cfg.get("lowercase_text", False),
                    normalize_whitespace_flag=processing_cfg.get("normalize_whitespace", True),
                )
                if normalized is None:
                    continue
                passages = build_passages_for_article(
                    normalized,
                    chunk_size=int(processing_cfg.get("chunk_words", 150)),
                    overlap=int(processing_cfg.get("overlap_words", 30)),
                    min_words=int(processing_cfg.get("min_words_per_passage", 40)),
                )
                for passage in passages:
                    writer.write(json.dumps(passage, ensure_ascii=False) + "\n")
                    passage_count += 1
                    file_passage_count += 1
        LOGGER.info("Finished %s passages_written=%d total_passages=%d", file_path.name, file_passage_count, passage_count)
    LOGGER.info("Saved wikipedia passages count=%d -> %s", passage_count, output_path)
    return output_path


def load_processed_qa_split(config: dict[str, Any], dataset_name: str, split_name: str) -> Path:
    qa_output_dir = config["processing"]["qa_output_dir"]
    split_file = qa_output_dir / f"{dataset_name}_{split_name}.jsonl"
    if not split_file.exists():
        raise FileNotFoundError(
            f"Processed QA split not found: {split_file}. Please run scripts/prepare_data.py first."
        )
    return split_file
