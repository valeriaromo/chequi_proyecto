from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

# PLACEHOLDER: reads src/app/data/mock/alerts.json.
# cuando las alertas estén dispo: derive items from monitor_rain.py output:
#  - estado == "RETIRAR" if skill_today < 0 and skill_90d_ago < 0
#  - estado == "ALERTA"  if skill_today < 0 only
#  - estado == "OK"      otherwise

_MOCK_PATH = Path(__file__).resolve().parent.parent / "data" / "mock" / "alerts.json"


@lru_cache(maxsize=1)
def _load_mock() -> dict[str, Any]:
    with _MOCK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def current_alerts() -> dict[str, Any]:
    return _load_mock()