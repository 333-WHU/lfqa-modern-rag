from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generation.cite_postprocess import extract_citations
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype config: {dtype_name}")
    return mapping[dtype_name]


class LocalGenerator:
    def __init__(self, generation_config: dict[str, Any], runtime_config: dict[str, Any]) -> None:
        model_path = generation_config["model_path"]
        if not model_path.exists():
            raise FileNotFoundError(f"Local generator model path not found: {model_path}")

        tokenizer_kwargs = {
            "local_files_only": runtime_config.get("local_files_only", True),
            "trust_remote_code": runtime_config.get("trust_remote_code", True),
        }
        model_kwargs: dict[str, Any] = {
            "local_files_only": runtime_config.get("local_files_only", True),
            "trust_remote_code": runtime_config.get("trust_remote_code", True),
            "device_map": runtime_config.get("device_map", "auto"),
            "torch_dtype": resolve_torch_dtype(runtime_config.get("torch_dtype", "auto")),
        }

        attn_impl = runtime_config.get("attn_implementation")
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        load_in_8bit = runtime_config.get("load_in_8bit", False)
        load_in_4bit = runtime_config.get("load_in_4bit", False)
        if load_in_8bit or load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover
                raise ImportError("bitsandbytes support requested but BitsAndBytesConfig is unavailable.") from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )

        LOGGER.info("Loading local generator model from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), **tokenizer_kwargs)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
        self.generation_config = generation_config

    def _build_input_text(self, prompt: str, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"

    def generate(self, prompt: str, system_prompt: str) -> dict[str, Any]:
        input_text = self._build_input_text(prompt, system_prompt=system_prompt)
        batch = self.tokenizer(input_text, return_tensors="pt")
        if not hasattr(self.model, "hf_device_map"):
            device = getattr(self.model, "device", None)
            if device is not None:
                batch = {key: value.to(device) for key, value in batch.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **batch,
                max_new_tokens=int(self.generation_config.get("max_new_tokens", 384)),
                temperature=float(self.generation_config.get("temperature", 0.7)),
                top_p=float(self.generation_config.get("top_p", 0.9)),
                do_sample=bool(self.generation_config.get("do_sample", True)),
                repetition_penalty=float(self.generation_config.get("repetition_penalty", 1.05)),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_length = batch["input_ids"].shape[-1]
        generated_ids = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return {
            "text": generated_text,
            "citations": extract_citations(generated_text),
        }
