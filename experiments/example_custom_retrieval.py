"""
Minimal example: use datasets module for cached embeddings,
plug in your own retrieval algorithm, evaluate with PyLate.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from tqdm import tqdm
from pylate import evaluation

from data import load


# ──────────────────────────────────────────────
# Step 1: Load dataset + embeddings (one line)
# ──────────────────────────────────────────────
ds = load("scifact")
print(ds)
# BEIRDataset(name='scifact', model='lightonai/GTE-ModernColBERT-v1',
#             docs=5183, queries=300, dim=128)


# ──────────────────────────────────────────────
# Step 2: YOUR ALGORITHM HERE
# ──────────────────────────────────────────────
def brute_force_maxsim(
    query_embs: list[np.ndarray],
    doc_embs: list[np.ndarray],
    doc_ids: list[str],
    k: int = 100,
) -> list[list[dict]]:
    """Exhaustive MaxSim: Score(Q, D) = sum_i max_j (Q_i · D_j)"""
    all_scores = []

    for q_emb in tqdm(query_embs, desc="Retrieving"):
        q = torch.from_numpy(q_emb).float().cuda()
        query_scores = []

        for d_emb in doc_embs:
            d = torch.from_numpy(d_emb).float().cuda()
            sim = q @ d.T
            score = sim.max(dim=1).values.sum().item()
            query_scores.append(score)

        top_indices = np.argsort(query_scores)[::-1][:k]
        all_scores.append([
            {"id": doc_ids[i], "score": query_scores[i]}
            for i in top_indices
        ])

    return all_scores


print("\nRunning brute-force retrieval...")
t0 = time.perf_counter()
scores = brute_force_maxsim(
    ds.query_embeddings, ds.doc_embeddings, ds.doc_ids, k=100,
)
elapsed = time.perf_counter() - t0
print(f"Done in {elapsed:.1f}s ({ds.num_queries / elapsed:.1f} QPS)")


# ──────────────────────────────────────────────
# Step 3: Evaluate (PyLate utility)
# ──────────────────────────────────────────────
results = evaluation.evaluate(
    scores=scores,
    qrels=ds.qrels,
    queries=ds.query_ids,
    metrics=["ndcg@10", "ndcg@100", "map", "recall@10", "recall@100"],
)

print(f"\n{'=' * 40}")
print(f"  {ds.name} Results (Brute-Force MaxSim)")
print(f"{'=' * 40}")
for metric, value in results.items():
    print(f"  {metric:>12s}: {value:.4f}")
