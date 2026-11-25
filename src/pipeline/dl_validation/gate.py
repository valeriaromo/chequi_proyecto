import json
import numpy as np
import cfg, common

def main():
    S = json.loads((cfg.OUT_DIR / "benchmark_summary.json").read_text())
    out = {}

    for t, by_cfg in S.items():
        best = max(by_cfg, key=lambda c: by_cfg[c]["mean"])
        b = by_cfg[best]
        entry = dict(best_config=best, dl_per_year=b["per_year"], dl_mean=b["mean"], seed_std=b["seed_std"])
        single, multi = f"{t}_only", "multitask"
        if single in by_cfg and multi in by_cfg:
            entry["multitask_minus_single"] = by_cfg[multi]["mean"] - by_cfg[single]["mean"]
        xgb = common.load_xgb(t)

        if xgb is None:
            out[t] = dict(entry, eligible=False, verdict="ADDITIONAL EXPERIMENTATION REQUIRED",
                          reason="no XGBoost baseline configured in cfg.BASELINE")
            continue

        years = list(b["per_year"])
        x = {y: float(xgb.get(int(y), np.nan)) for y in years}
        gain = np.array([b["per_year"][y] - x[y] for y in years])

        # determine if DL is recommended: mean gain > max(MIN_GAIN, NOISE_K * seed std) in every fold and DL skill positive in every fold
        thr = max(cfg.MIN_GAIN, cfg.NOISE_K * b["seed_std"])
        wins, dl_pos = int((gain > 0).sum()), all(v > 0 for v in b["per_year"].values())
        mean_gain = float(np.nanmean(gain))

        if mean_gain > thr and wins == len(gain) and dl_pos:
            verdict = "DEEP LEARNING RECOMMENDED"
        elif mean_gain <= thr and wins <= len(gain) // 2:
            verdict = "MACHINE LEARNING SUFFICIENT"
        else:
            verdict = "ADDITIONAL EXPERIMENTATION REQUIRED"
        out[t] = dict(entry, xgb_per_year=x, gain_per_year=dict(zip(years, gain.tolist())), mean_gain=mean_gain,
                      threshold=thr, wins=f"{wins}/{len(gain)}", dl_positive_all_years=dl_pos, verdict=verdict,
                      eligible=verdict == "DEEP LEARNING RECOMMENDED")
    (cfg.OUT_DIR / "gate.json").write_text(json.dumps(out, indent=2, default=float))
    print({t: g["verdict"] for t, g in out.items()})