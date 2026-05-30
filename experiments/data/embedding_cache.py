"""Flat array + offsets embedding cache with np.memmap support.

Storage layout per dataset:
    embeddings_cache/{name}/
    ├── doc_flat.npy       (N_total_docs, dim) float32 — all doc token vectors
    ├── doc_offsets.npy     (num_docs + 1,)    int64   — cumulative token counts
    ├── query_flat.npy      (N_total_queries, dim) float32
    └── query_offsets.npy   (num_queries + 1,) int64

Load uses np.memmap (mmap_mode="r") — zero-copy, OS pages in on demand.
"""

from __future__ import annotations

import os
import shutil

import numpy as np


def _cache_dir(
    dataset_name: str, model_name: str, cache_root: str, prefix: str = "",
) -> str:
    safe_model = model_name.replace("/", "_")
    safe_dataset = dataset_name.replace("/", "_")
    dir_name = f"{prefix}{safe_dataset}_{safe_model}"
    return os.path.join(cache_root, dir_name)


def save_embeddings(
    cache_path: str,
    doc_embeddings: list[np.ndarray],
    query_embeddings: list[np.ndarray],
) -> None:
    """Save variable-length embeddings as flat arrays + offsets."""
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    os.makedirs(cache_path)

    doc_flat = np.concatenate(doc_embeddings, axis=0).astype(np.float32)
    doc_lengths = [emb.shape[0] for emb in doc_embeddings]
    doc_offsets = np.zeros(len(doc_lengths) + 1, dtype=np.int64)
    np.cumsum(doc_lengths, out=doc_offsets[1:])

    query_flat = np.concatenate(query_embeddings, axis=0).astype(np.float32)
    query_lengths = [emb.shape[0] for emb in query_embeddings]
    query_offsets = np.zeros(len(query_lengths) + 1, dtype=np.int64)
    np.cumsum(query_lengths, out=query_offsets[1:])

    np.save(os.path.join(cache_path, "doc_flat.npy"), doc_flat)
    np.save(os.path.join(cache_path, "doc_offsets.npy"), doc_offsets)
    np.save(os.path.join(cache_path, "query_flat.npy"), query_flat)
    np.save(os.path.join(cache_path, "query_offsets.npy"), query_offsets)


def load_embeddings(
    cache_path: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load embeddings from flat + offsets cache using memmap."""
    doc_flat = np.load(
        os.path.join(cache_path, "doc_flat.npy"), mmap_mode="r",
    )
    doc_offsets = np.load(os.path.join(cache_path, "doc_offsets.npy"))

    query_flat = np.load(
        os.path.join(cache_path, "query_flat.npy"), mmap_mode="r",
    )
    query_offsets = np.load(os.path.join(cache_path, "query_offsets.npy"))

    doc_embeddings = [
        doc_flat[doc_offsets[i]:doc_offsets[i + 1]]
        for i in range(len(doc_offsets) - 1)
    ]
    query_embeddings = [
        query_flat[query_offsets[i]:query_offsets[i + 1]]
        for i in range(len(query_offsets) - 1)
    ]

    return doc_embeddings, query_embeddings


def cache_exists(cache_path: str) -> bool:
    """Check if a valid cache directory exists."""
    return (
        os.path.isdir(cache_path)
        and os.path.exists(os.path.join(cache_path, "doc_flat.npy"))
        and os.path.exists(os.path.join(cache_path, "doc_offsets.npy"))
        and os.path.exists(os.path.join(cache_path, "query_flat.npy"))
        and os.path.exists(os.path.join(cache_path, "query_offsets.npy"))
    )
