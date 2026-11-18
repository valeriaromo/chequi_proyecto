import json
import pandas as pd
import seg_config as cfg


def read_json(name):
    return json.loads((cfg.OUT / name).read_text(encoding="utf-8"))


def write_json(name, obj):
    text = json.dumps(obj, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    (cfg.OUT / name).write_text(text, encoding="utf-8")


def pc_cols(df):
    return [c for c in df.columns if c.startswith("PC")]


def load() -> pd.DataFrame:
    keys = ["municipio", "month"]
    raw = pd.read_parquet(cfg.OUT / "monthly_raw.parquet")
    scores = pd.read_parquet(cfg.OUT / "pca_scores.parquet")
    labels = pd.read_parquet(cfg.OUT / "labels.parquet")
    return raw.merge(scores, on=keys).merge(labels, on=keys)