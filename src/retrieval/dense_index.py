from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils.io import ensure_dir, iter_jsonl, write_json, write_jsonl
from src.utils.logger import get_logger

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError("faiss is required for dense indexing. Install faiss-cpu locally.") from exc


LOGGER = get_logger(__name__)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


class LocalTextEmbedder:
    def __init__(
        self,
        model_path: Path,
        batch_size: int = 16,
        max_length: int = 512,
        device: str | None = None,
        torch_dtype: str = "auto",
        low_cpu_mem_usage: bool = True,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Local embedding model path not found: {model_path}")
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.low_cpu_mem_usage = low_cpu_mem_usage
        LOGGER.info("Loading local embedding model from %s on device=%s", model_path, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
        )

        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }
        if self.device == "cuda":
            if torch_dtype == "auto":
                model_kwargs["torch_dtype"] = torch.float16
            elif torch_dtype == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            elif torch_dtype == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"

        self.model = AutoModel.from_pretrained(str(model_path), **model_kwargs)
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

        if hasattr(self.model, "hf_device_map"):
            self.input_device = None
        else:
            self.input_device = self.device

    def encode(self, texts: list[str], instruction: str = "") -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        prepared_texts = [f"{instruction} {text}".strip() if instruction else text for text in texts]
        batch = self.tokenizer(
            prepared_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if self.input_device is not None:
            batch = {key: value.to(self.input_device) for key, value in batch.items()}

        with torch.no_grad():
            model_output = self.model(**batch)
            pooled = mean_pool(model_output.last_hidden_state, batch["attention_mask"])
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.detach().cpu().numpy().astype(np.float32)


class DenseIndexer:
    def __init__(
        self,
        model_path: Path,
        output_dir: Path,
        batch_size: int = 16,
        max_length: int = 512,
        query_instruction: str = "Represent this question for retrieving supporting documents:",
        passage_instruction: str = "Represent this passage for retrieval:",
        torch_dtype: str = "auto",
        low_cpu_mem_usage: bool = True,
    ) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.embedder = LocalTextEmbedder(
            model_path=model_path,
            batch_size=batch_size,
            max_length=max_length,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        self.query_instruction = query_instruction
        self.passage_instruction = passage_instruction

    def build(self, passages_path: Path) -> dict[str, Path]:
        total_passages = count_jsonl_records(passages_path)
        if total_passages == 0:
            raise ValueError(f"No passages found in {passages_path}")

        index_path = self.output_dir / "passages.faiss"
        embeddings_path = self.output_dir / "passage_embeddings.npy"
        metadata_path = self.output_dir / "passages_metadata.jsonl"
        manifest_path = self.output_dir / "dense_manifest.json"

        if metadata_path.exists():
            metadata_path.unlink()
        if embeddings_path.exists():
            embeddings_path.unlink()
        if index_path.exists():
            index_path.unlink()

        index = None
        embedding_memmap = None
        write_offset = 0
        text_batch: list[str] = []
        metadata_batch: list[dict[str, Any]] = []

        with metadata_path.open("w", encoding="utf-8") as metadata_writer:
            for passage in tqdm(iter_jsonl(passages_path), total=total_passages, desc="Dense indexing"):
                text_batch.append(str(passage.get("text", "")))
                metadata_batch.append(passage)

                if len(text_batch) < self.embedder.batch_size:
                    continue

                batch_embeddings = self.embedder.encode(text_batch, instruction=self.passage_instruction)
                if batch_embeddings.ndim != 2:
                    raise RuntimeError("Dense embeddings must be 2D")

                if index is None:
                    dim = int(batch_embeddings.shape[1])
                    index = faiss.IndexFlatIP(dim)
                    embedding_memmap = np.lib.format.open_memmap(
                        embeddings_path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(total_passages, dim),
                    )

                index.add(batch_embeddings)
                embedding_memmap[write_offset : write_offset + len(batch_embeddings)] = batch_embeddings
                for item in metadata_batch:
                    metadata_writer.write(__import__("json").dumps(item, ensure_ascii=False) + "\n")
                write_offset += len(batch_embeddings)
                text_batch.clear()
                metadata_batch.clear()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if text_batch:
                batch_embeddings = self.embedder.encode(text_batch, instruction=self.passage_instruction)
                if index is None:
                    dim = int(batch_embeddings.shape[1])
                    index = faiss.IndexFlatIP(dim)
                    embedding_memmap = np.lib.format.open_memmap(
                        embeddings_path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(total_passages, dim),
                    )
                index.add(batch_embeddings)
                embedding_memmap[write_offset : write_offset + len(batch_embeddings)] = batch_embeddings
                for item in metadata_batch:
                    metadata_writer.write(__import__("json").dumps(item, ensure_ascii=False) + "\n")
                write_offset += len(batch_embeddings)

        if index is None or embedding_memmap is None:
            raise RuntimeError("Dense index build failed: no embeddings were created")

        embedding_memmap.flush()
        faiss.write_index(index, str(index_path))
        write_json(
            {
                "passages_path": str(passages_path),
                "total_passages": total_passages,
                "written_embeddings": write_offset,
                "embedding_dim": int(index.d),
                "batch_size": self.embedder.batch_size,
                "max_length": self.embedder.max_length,
            },
            manifest_path,
        )
        LOGGER.info("Built dense FAISS index at %s with passages=%d dim=%d", index_path, total_passages, index.d)
        return {
            "index": index_path,
            "embeddings": embeddings_path,
            "metadata": metadata_path,
            "manifest": manifest_path,
        }
