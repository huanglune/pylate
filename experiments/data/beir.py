"""
Cached BEIR dataset loader for vector set search experiments.

Usage:
    from beir_datasets import load, available

    ds = load("scifact")
    # ds.doc_embeddings:   list[np.ndarray], each (n_tokens, 128)
    # ds.query_embeddings: list[np.ndarray], each (n_tokens, 128)
    # ds.doc_ids:          list[str]
    # ds.query_ids:        list[str]
    # ds.qrels:            dict[query_id -> {doc_id: relevance}]

Available datasets (doc_length=300, ~200 tokens/doc avg):
    Name              Split  Queries    Corpus   Data   Est.Vectors  Encode.fp16  q_len
    ─────────────────────────────────────────────────────────────────────────────────
    nfcorpus          test       323      3,633    9MB        0.9M   0.2GB      32
    scifact           test       300      5,183    8MB        1.2M   0.3GB      48
    arguana           test     1,406      8,670   12MB        1.7M   0.4GB      64
    scidocs           test     1,000     25,000  251MB          5M   1.2GB      48
    fiqa              test       648     57,000   55MB         11M   2.8GB      32
    trec-covid        test        50    171,000  195MB         34M   8.5GB      48
    webis-touche2020  test        49    382,000  380MB         76M  18.5GB      32
    quora             test    10,000    523,000  210MB        105M  25.5GB      32
    nq                test     3,452  2,680,000  1.6GB       536M   131GB      32
    dbpedia-entity    test       400  4,630,000  1.8GB       926M   227GB      32
    hotpotqa          test     7,405  5,230,000  4.9GB      1.05B   256GB      32
    fever             test     6,666  5,420,000  3.9GB      1.08B   264GB      32
    climate-fever     test     1,535  5,420,000  3.9GB      1.08B   264GB      64
    msmarco           dev      6,980  8,840,000  7.3GB      1.77B   432GB      32

    Data   = raw dataset download size (corpus + queries + qrels).
    Est.Vectors = Corpus × ~200 tokens/doc (each token = one 128d vector).
    Encode.fp16 = estimated embedding cache size at fp16 (vectors × 128d × 2 bytes).
                  fp32 = 2× this value.
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

DATASET_CONFIGS: dict[str, dict] = {
    "scifact":           {"split": "test", "query_length": 48},
    "nfcorpus":          {"split": "test", "query_length": 32},
    "msmarco":           {"split": "dev",  "query_length": 32},
    "nq":                {"split": "test", "query_length": 32},
    "hotpotqa":          {"split": "test", "query_length": 32},
    "fiqa":              {"split": "test", "query_length": 32},
    "arguana":           {"split": "test", "query_length": 64},
    "scidocs":           {"split": "test", "query_length": 48},
    "trec-covid":        {"split": "test", "query_length": 48},
    "quora":             {"split": "test", "query_length": 32},
    "dbpedia-entity":    {"split": "test", "query_length": 32},
    "webis-touche2020":  {"split": "test", "query_length": 32},
    "fever":             {"split": "test", "query_length": 32},
    "climate-fever":     {"split": "test", "query_length": 64},
}

DEFAULT_MODEL = "lightonai/GTE-ModernColBERT-v1"
DEFAULT_DOC_LENGTH = 300
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "embeddings_cache")
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "evaluation_datasets")


@dataclass
class BEIRDataset:
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
            f"BEIRDataset(name={self.name!r}, model={self.model_name!r}, "
            f"docs={self.num_docs}, queries={self.num_queries}, "
            f"dim={self.embedding_dim})"
        )


def _get_cache_dir(dataset_name: str, model_name: str, cache_root: str, dtype: str = "fp16") -> str:
    return _cache_dir(dataset_name, model_name, cache_root, dtype=dtype)


def _load_raw(dataset_name: str, config: dict) -> tuple[list, dict, dict]:
    """Load raw BEIR data (documents, queries, qrels)."""
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    if "cqadupstack" in dataset_name:
        util.download_and_unzip(
            url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
            out_dir=DEFAULT_DATA_DIR,
        )
        data_path = os.path.join(DEFAULT_DATA_DIR, dataset_name)
    else:
        data_path = util.download_and_unzip(
            url=f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip",
            out_dir=DEFAULT_DATA_DIR,
        )

    # BEIR zips sometimes nest: out_dir/name/name/corpus.jsonl
    nested = os.path.join(data_path, dataset_name)
    if not os.path.exists(os.path.join(data_path, "corpus.jsonl")) and os.path.isdir(nested):
        data_path = nested

    documents, queries, qrels = GenericDataLoader(data_folder=data_path).load(
        split=config["split"],
    )
    documents = [
        {
            "id": doc_id,
            "text": f"{doc['title']} {doc['text']}".strip()
            if "title" in doc
            else doc["text"].strip(),
        }
        for doc_id, doc in documents.items()
    ]
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
) -> BEIRDataset:
    """Load a BEIR dataset with cached embeddings.

    First call encodes and caches to disk. Subsequent calls load from cache.

    Parameters
    ----------
    dataset_name
        BEIR dataset name (e.g. "scifact", "nfcorpus").
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

    documents, queries, qrels = _load_raw(dataset_name, config)
    doc_ids = [doc["id"] for doc in documents]
    query_ids = list(queries.keys())

    if not force_encode and cache_exists(cache_path):
        print(f"[beir] Loading cached embeddings ({dtype}): {cache_path}")
        doc_embeddings, query_embeddings = load_embeddings(cache_path)
    else:
        print(f"[beir] Encoding {dataset_name} with {model_name} ({dtype}) on {device or 'auto'} ...")
        doc_embeddings, query_embeddings = _encode(
            documents, queries, model_name, config["query_length"], doc_length,
            device=device, batch_size=batch_size,
        )
        save_embeddings(cache_path, doc_embeddings, query_embeddings, dtype)
        print(f"[beir] Cached to {cache_path}")

    return BEIRDataset(
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
    """List available dataset names."""
    return list(DATASET_CONFIGS.keys())
