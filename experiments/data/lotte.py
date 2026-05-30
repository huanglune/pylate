"""
Cached LoTTE dataset loader for vector set search experiments.

LoTTE (Long-Tail Topic-stratified Evaluation) — 5 domains × 2 query types.
Data source: mteb/LoTTE on HuggingFace.

Usage:
    from lotte_datasets import load, available

    ds = load("lifestyle_search")
    # Same interface as beir_datasets.BEIRDataset

Available datasets (from HuggingFace mteb/LoTTE, doc_length=300, ~200 tokens/doc avg):
    Name                  Queries    Corpus    Data   Est.Vectors  Encode.fp16  q_len
    ─────────────────────────────────────────────────────────────────────────────────
    lifestyle_search        1,080    388,000  ~400MB         78M    19GB      32
    lifestyle_forum         4,080    388,000  ~400MB         78M    19GB      32
    recreation_search       1,490    430,000  ~450MB         86M    21GB      32
    recreation_forum        4,000    430,000  ~450MB         86M    21GB      32
    writing_search          1,570    477,000  ~500MB         95M    23GB      32
    writing_forum           4,000    477,000  ~500MB         95M    23GB      32
    science_search          1,160  2,040,000  ~2.1GB        408M   100GB      32
    science_forum           4,030  2,040,000  ~2.1GB        408M   100GB      32
    technology_search       1,510  1,910,000  ~2.0GB        382M    93GB      32
    technology_forum        4,010  1,910,000  ~2.0GB        382M    93GB      32

    Data   = estimated HuggingFace download size.
    Est.Vectors = Corpus × ~200 tokens/doc (each token = one 128d vector).
    Encode.fp16 = estimated embedding cache size at fp16 (vectors × 128d × 2 bytes).
                  fp32 = 2× this value.
    Same domain shares corpus: search/forum only differ in queries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .embedding_cache import (
    _cache_dir,
    cache_exists,
    load_embeddings,
    save_embeddings,
)

LOTTE_DOMAINS = ["lifestyle", "recreation", "writing", "science", "technology"]
LOTTE_QUERY_TYPES = ["search", "forum"]

DATASET_CONFIGS: dict[str, dict] = {}
for _domain in LOTTE_DOMAINS:
    for _qtype in LOTTE_QUERY_TYPES:
        DATASET_CONFIGS[f"{_domain}_{_qtype}"] = {
            "domain": _domain,
            "query_type": _qtype,
            "query_length": 32,
        }

DEFAULT_MODEL = "lightonai/GTE-ModernColBERT-v1"
DEFAULT_DOC_LENGTH = 300
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "embeddings_cache")
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "evaluation_datasets", "lotte")


@dataclass
class LoTTEDataset:
    name: str
    model_name: str
    doc_ids: list[str]
    query_ids: list[str]
    qrels: dict
    doc_embeddings: list[np.ndarray]
    query_embeddings: list[np.ndarray]
    config: dict = field(default_factory=dict)

    @property
    def num_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def num_queries(self) -> int:
        return len(self.query_ids)

    @property
    def embedding_dim(self) -> int:
        return self.doc_embeddings[0].shape[1]

    def __repr__(self) -> str:
        return (
            f"LoTTEDataset(name={self.name!r}, model={self.model_name!r}, "
            f"docs={self.num_docs}, queries={self.num_queries}, "
            f"dim={self.embedding_dim})"
        )


def _get_cache_dir(dataset_name: str, model_name: str, cache_root: str, dtype: str = "fp16") -> str:
    return _cache_dir(dataset_name, model_name, cache_root, prefix="lotte_", dtype=dtype)


def _load_raw(config: dict) -> tuple[list, dict, dict]:
    """Load raw LoTTE data from HuggingFace (mteb/LoTTE)."""
    from datasets import load_dataset

    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)

    domain = config["domain"]
    qtype = config["query_type"]

    corpus_ds = load_dataset(
        "mteb/LoTTE", name=f"{domain}_{qtype}-corpus", split="test",
        cache_dir=DEFAULT_DATA_DIR,
    )
    queries_ds = load_dataset(
        "mteb/LoTTE", name=f"{domain}_{qtype}-queries", split="test",
        cache_dir=DEFAULT_DATA_DIR,
    )
    qrels_ds = load_dataset(
        "mteb/LoTTE", name=f"{domain}_{qtype}-qrels", split="test",
        cache_dir=DEFAULT_DATA_DIR,
    )

    documents = []
    for sample in corpus_ds:
        text = sample.get("text", "")
        title = sample.get("title", "")
        if title:
            text = f"{title} {text}".strip()
        documents.append({"id": str(sample["_id"]), "text": text})

    queries = {
        str(sample["_id"]): sample["text"]
        for sample in queries_ds
        if len(sample["text"]) > 0
    }

    qrels = {}
    for sample in qrels_ds:
        qid = str(sample["query-id"])
        did = str(sample["corpus-id"])
        score = int(sample["score"])
        if score > 0:
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = score

    return documents, queries, qrels


def _encode(
    documents: list,
    queries: dict,
    model_name: str,
    query_length: int,
    doc_length: int,
    device: str | None = None,
    batch_size: int = 256,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Encode documents and queries with ColBERT model."""
    from pylate import models

    kwargs = {}
    if device is not None:
        kwargs["device"] = device

    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=doc_length,
        query_length=query_length,
        **kwargs,
    )

    doc_embeddings = model.encode(
        sentences=[doc["text"] for doc in documents],
        batch_size=batch_size,
        is_query=False,
        show_progress_bar=True,
    )

    query_embeddings = model.encode(
        sentences=list(queries.values()),
        batch_size=min(batch_size, 64),
        is_query=True,
        show_progress_bar=True,
    )

    return doc_embeddings, query_embeddings


def load(
    dataset_name: str,
    model_name: str = DEFAULT_MODEL,
    doc_length: int = DEFAULT_DOC_LENGTH,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_encode: bool = False,
    dtype: str = "fp16",
    device: str | None = None,
    batch_size: int = 256,
) -> LoTTEDataset:
    """Load a LoTTE dataset with cached embeddings.

    Parameters
    ----------
    dataset_name
        LoTTE dataset name, e.g. "lifestyle_search", "science_forum".
    model_name
        HuggingFace model name for encoding.
    doc_length
        Max document token length.
    cache_dir
        Directory for embedding cache files.
    force_encode
        If True, re-encode even if cache exists.
    dtype
        Storage precision: "fp16" or "fp32".
    device
        Device for encoding (e.g. "cuda:0", "cuda:1"). None = auto-detect.
    batch_size
        Batch size for document encoding. Query batch size = min(batch_size, 64).
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset_name]
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _get_cache_dir(dataset_name, model_name, cache_dir, dtype)

    documents, queries, qrels = _load_raw(config)
    doc_ids = [doc["id"] for doc in documents]
    query_ids = list(queries.keys())

    if not force_encode and cache_exists(cache_path):
        print(f"[lotte] Loading cached embeddings ({dtype}): {cache_path}")
        doc_embeddings, query_embeddings = load_embeddings(cache_path)
    else:
        print(f"[lotte] Encoding {dataset_name} with {model_name} ({dtype}) on {device or 'auto'} ...")
        doc_embeddings, query_embeddings = _encode(
            documents, queries, model_name, config["query_length"], doc_length,
            device=device, batch_size=batch_size,
        )
        save_embeddings(cache_path, doc_embeddings, query_embeddings, dtype)
        print(f"[lotte] Cached to {cache_path}")

    return LoTTEDataset(
        name=dataset_name,
        model_name=model_name,
        doc_ids=doc_ids,
        query_ids=query_ids,
        qrels=qrels,
        doc_embeddings=doc_embeddings,
        query_embeddings=query_embeddings,
        config=config,
    )


def available() -> list[str]:
    """List available LoTTE dataset names."""
    return list(DATASET_CONFIGS.keys())
