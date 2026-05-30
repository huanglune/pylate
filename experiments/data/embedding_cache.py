"""Embedding cache with np.memmap support. Configurable precision (fp16/fp32).

Storage layout per dataset:
    embeddings_cache/{name}_fp{16|32}/
    ├── doc_encoding.npy      (N_total_docs, dim) float16 or float32
    ├── doc_doclens.npy       (num_docs,)         int32
    ├── query_encoding.npy    (N_total_queries, dim) float16 or float32
    └── query_doclens.npy     (num_queries,)      int32

Load uses np.memmap (mmap_mode="r") — zero-copy, OS pages in on demand.
"""

from __future__ import annotations

import os
import shutil

import numpy as np

DOC_ENCODING = "doc_encoding.npy"
DOC_DOCLENS = "doc_doclens.npy"
QUERY_ENCODING = "query_encoding.npy"
QUERY_DOCLENS = "query_doclens.npy"

DTYPE_MAP = {
    "fp16": np.float16,
    "fp32": np.float32,
}


def _cache_dir(
    dataset_name: str,
    model_name: str,
    cache_root: str,
    prefix: str = "",
    dtype: str = "fp16",
) -> str:
    safe_model = model_name.replace("/", "_")
    safe_dataset = dataset_name.replace("/", "_")
    dir_name = f"{prefix}{safe_dataset}_{safe_model}_{dtype}"
    return os.path.join(cache_root, dir_name)


def save_embeddings(
    cache_path: str,
    doc_embeddings: list[np.ndarray],
    query_embeddings: list[np.ndarray],
    dtype: str = "fp16",
) -> None:
    """Save variable-length embeddings as flat arrays + doclens."""
    np_dtype = DTYPE_MAP[dtype]

    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    os.makedirs(cache_path)

    doc_flat = np.concatenate(doc_embeddings, axis=0).astype(np_dtype)
    doc_doclens = np.array([emb.shape[0] for emb in doc_embeddings], dtype=np.int32)

    query_flat = np.concatenate(query_embeddings, axis=0).astype(np_dtype)
    query_doclens = np.array([emb.shape[0] for emb in query_embeddings], dtype=np.int32)

    np.save(os.path.join(cache_path, DOC_ENCODING), doc_flat)
    np.save(os.path.join(cache_path, DOC_DOCLENS), doc_doclens)
    np.save(os.path.join(cache_path, QUERY_ENCODING), query_flat)
    np.save(os.path.join(cache_path, QUERY_DOCLENS), query_doclens)


def load_embeddings(
    cache_path: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load embeddings from flat + doclens cache using memmap."""
    doc_flat = np.load(
        os.path.join(cache_path, DOC_ENCODING), mmap_mode="r",
    )
    doc_doclens = np.load(os.path.join(cache_path, DOC_DOCLENS))

    query_flat = np.load(
        os.path.join(cache_path, QUERY_ENCODING), mmap_mode="r",
    )
    query_doclens = np.load(os.path.join(cache_path, QUERY_DOCLENS))

    doc_offsets = np.zeros(len(doc_doclens) + 1, dtype=np.int64)
    np.cumsum(doc_doclens, out=doc_offsets[1:])

    query_offsets = np.zeros(len(query_doclens) + 1, dtype=np.int64)
    np.cumsum(query_doclens, out=query_offsets[1:])

    doc_embeddings = [
        doc_flat[doc_offsets[i]:doc_offsets[i + 1]]
        for i in range(len(doc_doclens))
    ]
    query_embeddings = [
        query_flat[query_offsets[i]:query_offsets[i + 1]]
        for i in range(len(query_doclens))
    ]

    return doc_embeddings, query_embeddings


def cache_exists(cache_path: str) -> bool:
    """Check if a valid cache directory exists."""
    return (
        os.path.isdir(cache_path)
        and os.path.exists(os.path.join(cache_path, DOC_ENCODING))
        and os.path.exists(os.path.join(cache_path, DOC_DOCLENS))
        and os.path.exists(os.path.join(cache_path, QUERY_ENCODING))
        and os.path.exists(os.path.join(cache_path, QUERY_DOCLENS))
    )
