from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

# PLACEHOLDER: reads src/app/data/mock/metrics.json, 
#TODO(modelo): read `outputs_fase2/rain_eval.csv` and
#`outputs_fase2/scope_<variable>.csv` and aggregate per horizon.

_MOCK_PATH = Path(__file__).resolve().parent.parent / "data" / "mock" / "metrics.json"


@lru_cache(maxsize=1)
def _load_mock() -> dict[str, Any]:
    with _MOCK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def summary() -> dict[str, Any]:
    return _load_mock()