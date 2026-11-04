from __future__ import annotations
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# metrics.py -> services -> app -> src
DATA_ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "data_analysis" / "output" / "data_analysis"
_MOCK_PATH = Path(__file__).resolve().parent.parent / "data" / "mock" / "metrics.json"


@lru_cache(maxsize=4)
def pca_scatter(perfil: str = "sin_aod", n: int = 1500) -> dict[str, Any]:
    tab = DATA_ANALYSIS_DIR / perfil / "tables"
    scores = pd.read_parquet(tab / "pca_scores.parquet")
    summ = pd.read_csv(tab / "step3_pca_summary.csv")
    s = scores.sample(min(n, len(scores)), random_state=42)
    return {
        "explained_variance": (summ["Explained_Variance_%"].iloc[:2] / 100).round(4).tolist(),
        "points": [{"pc1": float(a), "pc2": float(b), "label": m}
                   for a, b, m in zip(s.PC1, s.PC2, s.municipio)],
    }


@lru_cache(maxsize=1)
def _load_mock() -> dict[str, Any]:
    with _MOCK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def summary() -> dict[str, Any]:
    data = dict(_load_mock())
    try:
        data["pca_scatter"] = pca_scatter()
    except Exception:
        log.exception("No se pudo construir pca_scatter desde %s", DATA_ANALYSIS_DIR)
    return data