# LFQA Modern RAG

An offline-first, locally runnable LFQA / RAG project for reproducing and extending long-form question answering systems.

This project is designed around strict local-only constraints:

- load data only from local disk
- load models only from local directories
- never call `load_dataset("...")`
- never call `snapshot_download` or `hf_hub_download`
- use `local_files_only=True` when supported by Transformers loaders
- keep paths and key hyperparameters inside `configs/*.yaml`

## Current mainline

The current resource-friendly mainline is:

- first-stage retrieval: BM25
- second-stage retrieval: reranker
- generation: local Qwen instruct model

Dense full-corpus indexing is kept in the codebase for reference, but it is not part of the default mainline because it is too memory-heavy for large Wikipedia-scale corpora on many single-node environments.

## Local resource layout

Prepare the following directories before running the project:

- local QA data
- local Wikipedia or wiki snippets data
- local reranker model
- local generator model

Default paths live in `configs/data.yaml`, `configs/retrieval.yaml`, `configs/generation.yaml`, and `configs/eval.yaml`.

## Supported data formats

The local loaders support:

- JSONL
- JSON
- CSV
- Parquet

Common patterns:

- split files such as `train.jsonl`, `valid.jsonl`, `test.jsonl`
- a directory with multiple `json/jsonl/csv/parquet` files
- Wikipedia dumps stored as multiple local files

If your field names differ, adapt `field_map` in `configs/data.yaml`.

## Run order

Run from the project root:

```bash
python scripts/prepare_data.py
python scripts/build_indexes.py
python scripts/run_baseline.py
python scripts/run_full_pipeline.py
```

## Script summary

- `prepare_data.py`
  - loads local ELI5 / Wikipedia / ASQA
  - normalizes QA examples
  - slices Wikipedia into passages
- `build_indexes.py`
  - builds a BM25 index only
- `run_baseline.py`
  - uses BM25 retrieval only
  - skips reranking
  - generates answers directly
- `run_full_pipeline.py`
  - uses BM25 first-stage retrieval
  - applies reranking as second-stage retrieval
  - uses citation-aware prompting
  - writes generation outputs and evaluation metrics

## Output locations

- passages: `data/processed/wiki_passages/passages.jsonl`
- retrieval results: `outputs/retrieval_results/`
- generations: `outputs/generations/`
- metrics: `outputs/metrics/`

## Notes

- The code assumes all local models and datasets are already prepared.
- The project does not trigger or recommend any online download commands.
- If `rouge-score` is unavailable, ROUGE-L evaluation is skipped with a clear log message.
- If you move between Windows and Linux cloud environments, update the local paths in the YAML configs to match the current machine.
