"""Dataset loading utilities for vector set search experiments.

Unified interface:
    from data import load

    ds = load("scifact")           # BEIR dataset
    ds = load("lifestyle_search")  # LoTTE dataset

Or load from specific sources:
    from data import beir, lotte

    ds = beir.load("scifact")
    ds = lotte.load("lifestyle_search")
"""

from . import beir, lotte


def load(dataset_name: str, **kwargs):
    """Auto-detect dataset source (BEIR or LoTTE) and load with cached embeddings."""
    if dataset_name in beir.DATASET_CONFIGS:
        return beir.load(dataset_name, **kwargs)
    if dataset_name in lotte.DATASET_CONFIGS:
        return lotte.load(dataset_name, **kwargs)
    raise ValueError(
        f"Unknown dataset {dataset_name!r}. "
        f"BEIR: {beir.available()}, "
        f"LoTTE: {lotte.available()}"
    )


def available() -> dict[str, list[str]]:
    """List all available dataset names by source."""
    return {
        "beir": beir.available(),
        "lotte": lotte.available(),
    }
