from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [make_json_serializable(item) for item in value.tolist()]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return make_json_serializable(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_serializable(item) for item in value]
    return str(value)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: Path, indent: int = 2) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_json_serializable(data), f, ensure_ascii=False, indent=indent)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected dict record in {path} at line {line_number}")
            yield make_json_serializable(record)


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(make_json_serializable(record), ensure_ascii=False) + "\n")


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def read_parquet_records(path: Path) -> list[dict[str, Any]]:
    if pd is None:
        raise ImportError(f"Reading parquet requires pandas and pyarrow. Missing dependency for {path}")
    dataframe = pd.read_parquet(path)
    return [make_json_serializable(record) for record in dataframe.to_dict(orient="records")]


def iter_parquet_records(path: Path, batch_size: int = 5000) -> Iterator[dict[str, Any]]:
    if pq is None:
        for record in read_parquet_records(path):
            yield record
        return
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for record in batch.to_pylist():
            if isinstance(record, dict):
                yield make_json_serializable(record)


def dump_pickle(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(data, f)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def iter_records_from_file(path: Path, parquet_batch_size: int = 5000) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from iter_jsonl(path)
        return
    if suffix == ".json":
        try:
            payload = read_json(path)
        except json.JSONDecodeError:
            yield from iter_jsonl(path)
            return
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield make_json_serializable(item)
            return
        if isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], list):
                for item in payload["data"]:
                    if isinstance(item, dict):
                        yield make_json_serializable(item)
                return
            yield make_json_serializable(payload)
            return
        raise ValueError(f"Unsupported JSON payload type in {path}: {type(payload)!r}")
    if suffix == ".csv":
        for record in read_csv_records(path):
            yield make_json_serializable(record)
        return
    if suffix == ".parquet":
        yield from iter_parquet_records(path, batch_size=parquet_batch_size)
        return
    raise ValueError(f"Unsupported file type: {path}")


def load_records_from_file(path: Path) -> list[dict[str, Any]]:
    return list(iter_records_from_file(path))
