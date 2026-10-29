from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_MOCK_PATH = Path(__file__).resolve().parent.parent / "data" / "mock" / "functionality.json"


@lru_cache(maxsize=1)
def _load() -> list[dict[str, Any]]:
    with _MOCK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def all_blocks() -> list[dict[str, Any]]:
    return _load()


def get_block(slug: str) -> dict[str, Any] | None:
    return next((b for b in _load() if b["slug"] == slug), None)