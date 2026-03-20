from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


CONFIG_FILES = ("data.yaml", "retrieval.yaml", "generation.yaml", "eval.yaml")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def deep_merge(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in new.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def infer_project_root(configured_root: str | None, cwd: Path) -> Path:
    if not configured_root:
        return cwd.resolve()

    raw_path = Path(configured_root)
    if raw_path.is_absolute():
        return raw_path.resolve()

    candidate = (cwd / raw_path).resolve()
    if candidate.exists():
        return candidate

    if raw_path.name and raw_path.name == cwd.name:
        return cwd.resolve()

    return candidate


def resolve_path(path_value: str | None, project_root: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_project_config(config_dir: Path | None = None, cwd: Path | None = None) -> dict[str, Any]:
    current_dir = cwd.resolve() if cwd is not None else Path.cwd().resolve()
    config_root = config_dir.resolve() if config_dir is not None else current_dir / "configs"
    if not config_root.exists():
        raise FileNotFoundError(f"Config directory not found: {config_root}")

    config: dict[str, Any] = {}
    for name in CONFIG_FILES:
        file_path = config_root / name
        if not file_path.exists():
            raise FileNotFoundError(f"Required config file missing: {file_path}")
        config = deep_merge(config, load_yaml(file_path))

    project_cfg = config.get("project", {})
    project_root = infer_project_root(project_cfg.get("project_root"), current_dir)
    config["project"] = {"project_root": project_root}

    path_cfg = config.setdefault("paths", {})
    for key, value in list(path_cfg.items()):
        path_cfg[key] = resolve_path(value, project_root)

    retrieval_index = config.setdefault("index", {})
    for key in ("bm25_dir", "faiss_dir"):
        if key in retrieval_index:
            retrieval_index[key] = resolve_path(retrieval_index[key], project_root)

    processing_cfg = config.setdefault("processing", {})
    for key in ("qa_output_dir", "wiki_passages_output"):
        if key in processing_cfg:
            processing_cfg[key] = resolve_path(processing_cfg[key], project_root)

    generation_cfg = config.setdefault("generation", {})
    if "model_path" in generation_cfg:
        generation_cfg["model_path"] = resolve_path(generation_cfg["model_path"], project_root)

    model_cfg = config.setdefault("models", {})
    for key, value in list(model_cfg.items()):
        model_cfg[key] = resolve_path(value, project_root)

    return config
