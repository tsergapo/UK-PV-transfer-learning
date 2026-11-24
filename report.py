"""report.py - which differences between settings are real, and which are noise?

    python report.py                    (reads results/dev/daily_errors.csv)
    python report.py --phase final
    python report.py --a xgb/finetune_12m --b xgb/scratch_12m      (one comparison of your own)

With 10 target sites and one year, two settings that differ by 0.003 in RMSE may be equal. This
script repeats the scoring 2000 times on resampled data and shows how much the numbers move:

  * resampling DAYS    : would the result change with different weather?
  * resampling SITES   : would the result change with different target roofs? (the larger
                         uncertainty, because there are only 10 of them)

A difference is called clear only if its 95% interval excludes 0 in BOTH.
Errors are daytime RMSE in kWh per kWp per hour; "a - b" below 0 means a is better.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# comparisons printed by default: (a, b), written model/setting
DEFAULT_PAIRS = [
    ("xgb/rolling_7d", "xgb/zero_shot"),
    ("xgb/calibrated_1m", "xgb/scratch_1m"),
    ("xgb/calibrated_3m", "xgb/scratch_3m"),
    ("xgb/finetune_1m", "xgb/calibrated_1m"),
    ("xgb/finetune_12m", "xgb/scratch_12m"),
    ("xgb/finetune_12m", "xgb/rolling_7d"),
    ("xgb/scratch_12m", "xgb/rolling_7d"),
    ("xgb/rolling_7d", "day_before/none"),
]


def prepare(daily: pd.DataFrame, role: str = "target") -> pd.DataFrame:
    d = daily[daily["role"] == role].copy()
    d["key"] = d["model"] + "/" + d["setting"].astype(str)
    return d


def _sums(d: pd.DataFrame, key: str, by: str, units) -> tuple[np.ndarray, np.ndarray]:
    g = d[d["key"] == key].groupby(by)[["sse", "n"]].sum().reindex(units, fill_value=0)
    return g["sse"].to_numpy(dtype=float), g["n"].to_numpy(dtype=float)


def bootstrap(d: pd.DataFrame, keys: list[str], by: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Pooled daytime RMSE of each key on `n_boot` resamples of the units in column `by`
    ('date' or 'ss_id'). The same resamples are used for every key, so differences are paired."""
    units = np.sort(d[by].unique())
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(units), np.full(len(units), 1.0 / len(units)), size=n_boot)   # how often each unit is drawn
    out = {}
    for k in keys:
        sse, n = _sums(d, k, by, units)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = np.sqrt((w @ sse) / (w @ n))
    return out


def point(d: pd.DataFrame, key: str) -> float:
    s = d[d["key"] == key][["sse", "n"]].sum()
    return float(np.sqrt(s["sse"] / s["n"]))


def interval(x: np.ndarray) -> tuple[float, float]:
    return tuple(np.nanpercentile(x, [2.5, 97.5]))


def site_wins(d: pd.DataFrame, a: str, b: str) -> tuple[int, int]:
    """On how many sites is a better (lower RMSE) than b, out of the sites both were scored on."""
    def per_site(k):
        g = d[d["key"] == k].groupby("ss_id")[["sse", "n"]].sum()
        return np.sqrt(g["sse"] / g["n"])
    both = pd.concat([per_site(a), per_site(b)], axis=1, keys=["a", "b"]).dropna()
    return int((both["a"] < both["b"]).sum()), len(both)


def compare(d: pd.DataFrame, a: str, b: str, n_boot: int = 2000) -> dict:
    by_day, by_site = (bootstrap(d, [a, b], by, n_boot) for by in ("date", "ss_id"))
    diff = point(d, a) - point(d, b)
    lo_d, hi_d = interval(by_day[a] - by_day[b])
    lo_s, hi_s = interval(by_site[a] - by_site[b])
    clear = (hi_d < 0 and hi_s < 0) or (lo_d > 0 and lo_s > 0)
    wins, total = site_wins(d, a, b)
    return {"a": a, "b": b, "diff": diff, "days_95": (lo_d, hi_d), "sites_95": (lo_s, hi_s),
            "a_better_on_sites": f"{wins}/{total}",
            "verdict": ("clear: " + ("a better" if diff < 0 else "b better")) if clear else "not clear"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Uncertainty of the experiment's results.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--phase", default="dev")
    parser.add_argument("--a")
    parser.add_argument("--b")
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()
    daily = pd.read_csv(Path(args.results) / args.phase / "daily_errors.csv", parse_dates=["date"])
    d = prepare(daily)

    keys = sorted(d["key"].unique(), key=lambda k: point(d, k))
    by_day, by_site = (bootstrap(d, keys, by, args.boot) for by in ("date", "ss_id"))
    rows = [{"setting": k, "rmse_day": point(d, k),
             "range_if_other_weather": "{:.4f} - {:.4f}".format(*interval(by_day[k])),
             "range_if_other_sites": "{:.4f} - {:.4f}".format(*interval(by_site[k]))} for k in keys]
    print(f"TARGET sites, {args.phase} run. 95% ranges from {args.boot} resamples.\n")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    pairs = [(args.a, args.b)] if args.a and args.b else DEFAULT_PAIRS
    print("\nDifferences (a - b; below 0 means a is better):\n")
    for a, b in pairs:
        if a not in keys or b not in keys:
            print(f"  {a} vs {b}: not in the results, skipped")
            continue
        c = compare(d, a, b, args.boot)
        print(f"  {a} vs {b}: {c['diff']:+.4f}   weather {c['days_95'][0]:+.4f}..{c['days_95'][1]:+.4f}"
              f"   sites {c['sites_95'][0]:+.4f}..{c['sites_95'][1]:+.4f}   a better on {c['a_better_on_sites']} sites"
              f"   -> {c['verdict']}")


if __name__ == "__main__":
    main()
