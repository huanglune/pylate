"""Run benchmark evaluation with configurable index backends and datasets.

Supports both BEIR and LoTTE datasets. Dataset names are auto-detected:
BEIR names (scifact, nfcorpus, ...) go through beir_datasets,
LoTTE names (lifestyle_search, science_forum, ...) go through lotte_datasets.
See beir_datasets.py and lotte_datasets.py for full dataset tables.

Usage:
    python run_beir_benchmark.py                                       # default
    python run_beir_benchmark.py --index plaid warp                    # multiple indexes
    python run_beir_benchmark.py --datasets scifact lifestyle_search   # mix BEIR + LoTTE
    python run_beir_benchmark.py --index warp --datasets scifact nfcorpus

Available indexes:
    plaid       IVF+PQ, centroid interaction (CIKM 2022). Retriever: ColBERT MaxSim.
    warp        IVF+PQ, implicit decompression (SIGIR 2025). Retriever: XTR imputation.

Available datasets:
    BEIR:  nfcorpus, scifact, arguana, scidocs, fiqa, trec-covid,
           webis-touche2020, quora, nq, dbpedia-entity, hotpotqa,
           fever, climate-fever, msmarco
    LoTTE: lifestyle_search, lifestyle_forum, recreation_search,
           recreation_forum, writing_search, writing_forum,
           science_search, science_forum, technology_search, technology_forum
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from pylate import evaluation, indexes, retrieve
from tqdm import tqdm

from data import load as load_dataset

METRICS = [
    "ndcg@1", "ndcg@5", "ndcg@10", "ndcg@100",
    "map",
    "recall@10", "recall@100",
    "hits@1", "hits@5", "hits@10",
]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def build_index_and_retriever(
    index_type: str, dataset_name: str,
) -> tuple:
    if index_type == "plaid":
        index = indexes.PLAID(
            override=True,
            index_name=f"benchmark_{dataset_name}_plaid",
            show_progress=False,
        )
        retriever = retrieve.ColBERT(index=index)
    elif index_type == "warp":
        index = indexes.WARP(
            override=True,
            index_name=f"benchmark_{dataset_name}_warp",
            show_progress=False,
        )
        retriever = retrieve.XTR(index=index)
    else:
        raise ValueError(f"Unknown index type: {index_type!r}. Use 'plaid' or 'warp'.")
    return index, retriever


def run_single_dataset(dataset_name: str, index_type: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}  |  Index: {index_type.upper()}")
    print(f"{'='*60}")

    ds = load_dataset(dataset_name)
    print(ds)

    index, retriever = build_index_and_retriever(index_type, dataset_name)

    print(f"[1/3] Building {index_type.upper()} index...")
    t0 = time.perf_counter()
    index.add_documents(
        documents_ids=ds.doc_ids,
        documents_embeddings=ds.doc_embeddings,
    )
    index_time = time.perf_counter() - t0
    print(f"       Done in {index_time:.1f}s")

    print("[2/3] Retrieving (per-query)...")
    scores = []
    per_query_latencies = []
    for q_emb in tqdm(ds.query_embeddings, desc="Retrieving queries"):
        t0 = time.perf_counter()
        result = retriever.retrieve(queries_embeddings=[q_emb], k=100)
        per_query_latencies.append(time.perf_counter() - t0)
        scores.extend(result)

    latency_ms = np.array(per_query_latencies) * 1000
    retrieve_time = sum(per_query_latencies)
    qps = ds.num_queries / retrieve_time
    print(f"       {ds.num_queries} queries in {retrieve_time:.2f}s ({qps:.1f} QPS)")

    print("[3/3] Computing metrics...")
    eval_scores = evaluation.evaluate(
        scores=scores,
        qrels=ds.qrels,
        queries=ds.query_ids,
        metrics=METRICS,
    )

    result = {
        "dataset": dataset_name,
        "model": ds.model_name,
        "index": index_type.upper(),
        "corpus_size": ds.num_docs,
        "num_queries": ds.num_queries,
        "metrics": eval_scores,
        "timing": {
            "build_index_sec": round(index_time, 2),
            "retrieve_sec": round(retrieve_time, 2),
            "queries_per_second": round(qps, 1),
            "latency_ms": {
                "mean": round(float(latency_ms.mean()), 2),
                "p50": round(float(np.percentile(latency_ms, 50)), 2),
                "p95": round(float(np.percentile(latency_ms, 95)), 2),
                "p99": round(float(np.percentile(latency_ms, 99)), 2),
                "min": round(float(latency_ms.min()), 2),
                "max": round(float(latency_ms.max()), 2),
            },
        },
    }

    print(f"\n  Results for {dataset_name} ({index_type.upper()}):")
    for k, v in eval_scores.items():
        print(f"    {k:>15s}: {v:.4f}")
    print(f"    {'QPS':>15s}: {qps:.1f}")
    lat = result["timing"]["latency_ms"]
    print(f"    {'latency(ms)':>15s}: mean={lat['mean']:.1f}  "
          f"p50={lat['p50']:.1f}  p95={lat['p95']:.1f}  p99={lat['p99']:.1f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark with configurable indexes and datasets (BEIR + LoTTE)",
    )
    parser.add_argument(
        "--index", nargs="+", default=["plaid", "warp"],
        help="Index backends to benchmark (default: plaid warp)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["scifact", "nfcorpus"],
        help="Datasets to evaluate, supports both BEIR and LoTTE names (default: scifact nfcorpus)",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for idx_type in args.index:
        for name in args.datasets:
            key = f"{name}/{idx_type}"
            all_results[key] = run_single_dataset(name, idx_type)

    output_path = os.path.join(RESULTS_DIR, "benchmark.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    header = (f"{'Dataset':>20s}  {'Index':>6s}  {'ndcg@10':>8s}  {'recall@100':>10s}  "
              f"{'map':>6s}  {'QPS':>6s}  {'p50ms':>6s}  {'p95ms':>6s}  {'p99ms':>6s}")
    print(header)

    for key, r in all_results.items():
        lat = r["timing"]["latency_ms"]
        row = (f"{r['dataset']:>20s}  {r['index']:>6s}  {r['metrics']['ndcg@10']:>8.4f}  "
               f"{r['metrics']['recall@100']:>10.4f}  {r['metrics']['map']:>6.4f}  "
               f"{r['timing']['queries_per_second']:>6.1f}  "
               f"{lat['p50']:>6.1f}  {lat['p95']:>6.1f}  {lat['p99']:>6.1f}")
        print(row)


if __name__ == "__main__":
    main()
