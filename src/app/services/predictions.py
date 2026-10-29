from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

# PLACEHOLDER: reads src/app/data/mock/predictions.json.
# cuando el modelo de predicción esté listo : replace `_load_mock()` with a real adapter that:
#  - loads `outputs_fase2/rain_eval.csv` and `scope_T2M_MIN.csv`
#  - loads the frozen models referenced in `monitor_rain.py`
#  - returns the same dictionary shape (see below).

_MOCK_PATH = Path(__file__).resolve().parent.parent / "data" / "mock" / "predictions.json"


@lru_cache(maxsize=1)
def _load_mock() -> dict[str, Any]:
    with _MOCK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def list_municipalities() -> list[str]:
    return list(_load_mock()["by_municipality"].keys())


def predict(municipio: str) -> dict[str, Any]:
    data = _load_mock()["by_municipality"]
    return data.get(municipio, data["Aguascalientes"])


def snapshot() -> dict[str, Any]:
    return _load_mock()