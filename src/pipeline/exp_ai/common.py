import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cfg


def make_dirs():
    for d in (cfg.IMG, cfg.JS, cfg.MODELS):
        d.mkdir(parents=True, exist_ok=True)


def _enc(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return str(o)


def save_json(obj, name):
    make_dirs()
    (cfg.JS / name).write_text(json.dumps(obj, indent=2, default=_enc, ensure_ascii=False), encoding="utf-8")


def load_json(name):
    return json.loads((cfg.JS / name).read_text(encoding="utf-8"))


def save_fig(name):
    make_dirs()
    plt.gcf().savefig(cfg.IMG / name, dpi=130, bbox_inches="tight")
    plt.close("all")
    return f"images/{name}"


def add_viz(target, title, path, what, obs, practical):
    f = cfg.JS / "dashboard.json"
    items = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
    items = [i for i in items if not (i["target"] == target and i["title"] == title)]
    items.append(dict(target=target, title=title, path=path, what=what, obs=obs, practical=practical))
    save_json(items, "dashboard.json")


def load_model(name):
    return joblib.load(cfg.MODELS / f"{name}.joblib")


def predict(model, X, kind):
    return model.predict_proba(X)[:, 1] if kind == "clf" else model.predict(X)


def md_table(headers, rows):
    clean = lambda v: str(v).replace("|", "/").replace("\n", " ")
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(clean(r.get(h, "")) for h in headers) + " |" for r in rows]
    return "\n".join(out)


def select_cases(pred, y, kind):
    """Representative (typical, correct) and critical (uncertain / wrong) cases -> [(group, label, pos)]."""
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    if kind == "clf":
        ok = (pred >= cfg.THR) == (y == 1)
    else:
        ok = np.abs(pred - y) <= np.median(np.abs(pred - y))
    cand = np.where(ok)[0]
    cases = []
    for q in (0.25, 0.75):
        t = np.quantile(pred[cand], q)
        cases.append(("representative", f"typical_p{int(q * 100)}", cand[np.argmin(np.abs(pred[cand] - t))]))
    if kind == "clf":
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        cases += [("critical", "most_uncertain", np.argmin(np.abs(pred - cfg.THR))),
                  ("critical", "missed_event", pos[np.argmin(pred[pos])]),
                  ("critical", "false_alarm", neg[np.argmax(pred[neg])])]
    else:
        err = pred - y
        cases += [("critical", "largest_overestimate", np.argmax(err)),
                  ("critical", "largest_underestimate", np.argmin(err))]
    return [(g, l, int(i)) for g, l, i in cases]