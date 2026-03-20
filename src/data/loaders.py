from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.io import load_records_from_file
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)
SUPPORTED_SUFFIXES = (".jsonl", ".json", ".csv", ".parquet")


@dataclass
class LoadedSplit:
    name: str
    records: list[dict[str, Any]]


def _candidate_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file format: {path}")
        return [path]

    files = [file for file in path.rglob("*") if file.is_file() and file.suffix.lower() in SUPPORTED_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"No supported data files found under {path}")
    return sorted(files)


def _match_split(file_path: Path, split_aliases: dict[str, list[str]]) -> str | None:
    file_name = file_path.stem.lower()
    parent_name = file_path.parent.name.lower()
    for split_name, aliases in split_aliases.items():
        for alias in aliases:
            alias_lower = alias.lower()
            if alias_lower in file_name or alias_lower in parent_name:
                return split_name
    return None


def _extract_from_dict_payload(
    payload: dict[str, Any], split_aliases: dict[str, list[str]]
) -> dict[str, list[dict[str, Any]]]:
    split_records: dict[str, list[dict[str, Any]]] = {}
    for split_name, aliases in split_aliases.items():
        for alias in [split_name, *aliases]:
            if alias in payload and isinstance(payload[alias], list):
                split_records[split_name] = [item for item in payload[alias] if isinstance(item, dict)]
                break
    return split_records


class LocalDatasetLoader:
    """Robust local file loader for QA and wiki datasets."""

    def __init__(self, dataset_name: str, dataset_config: dict[str, Any], path: Path) -> None:
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.path = path

    def load(self) -> dict[str, list[dict[str, Any]]]:
        if not self.path.exists():
            raise FileNotFoundError(f"Local dataset path does not exist: {self.path}")

        split_aliases = self.dataset_config.get("split_aliases", {})
        if self.path.is_file() and self.path.suffix.lower() == ".json":
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    split_records = _extract_from_dict_payload(payload, split_aliases)
                    if split_records:
                        LOGGER.info("Loaded %s split payloads directly from JSON file %s", self.dataset_name, self.path)
                        return split_records
            except json.JSONDecodeError:
                pass

        split_buckets: dict[str, list[dict[str, Any]]] = {}
        for file_path in _candidate_files(self.path):
            split_name = _match_split(file_path, split_aliases) or "all"
            records = load_records_from_file(file_path)
            split_buckets.setdefault(split_name, []).extend(records)

        if not split_buckets:
            raise RuntimeError(f"No records loaded for dataset {self.dataset_name} from {self.path}")

        for split_name, records in split_buckets.items():
            LOGGER.info(
                "Loaded dataset=%s split=%s records=%d from %s",
                self.dataset_name,
                split_name,
                len(records),
                self.path,
            )
        return split_buckets


def get_field_value(record: Any, field: str) -> Any:
    parts = field.split(".") if field else []
    return _get_nested_value(record, parts)


def _get_nested_value(current: Any, parts: list[str]) -> Any:
    if not parts:
        if hasattr(current, "tolist") and not isinstance(current, (str, bytes, bytearray)):
            try:
                return current.tolist()
            except Exception:
                return current
        return current

    if hasattr(current, "tolist") and not isinstance(current, (str, bytes, bytearray)):
        try:
            current = current.tolist()
        except Exception:
            pass

    key = parts[0]
    rest = parts[1:]

    if isinstance(current, dict):
        if key not in current:
            return None
        return _get_nested_value(current[key], rest)

    if isinstance(current, (list, tuple)):
        values: list[Any] = []
        for item in current:
            value = _get_nested_value(item, parts)
            if value is None:
                continue
            if isinstance(value, list):
                values.extend(value)
            else:
                values.append(value)
        return values if values else None

    return None


def pick_first(record: dict[str, Any], candidate_fields: list[str]) -> Any:
    for field in candidate_fields:
        value = get_field_value(record, field)
        if value not in (None, "", []):
            return value
    return None
