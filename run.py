"""run.py - the whole experiment in one command.

    python run.py                 development run: trained before the validation year, scored ON the validation year
    python run.py --final         final run: trained before test_start, scored on the TEST period (do this once,
                                  after everything else is decided)
    python run.py --quick         5 sources and 2 targets, to check that everything works in a minute

What it compares, for every target site (sites never trained on), scored on the same days:

    day_before               baseline: yesterday's output at the same hour
    linear_ssrd, xgb
      zero_shot              trained on the 30 source sites only; applied to the target as it is
      rolling_7d             zero_shot x a factor re-fitted each day on the site's previous 7 days
      calibrated_<N>m        zero_shot x ONE factor fitted on the target's last N months
      scratch_<N>m           trained on the target's own last N months only (no source sites)
      finetune_<N>m          the source model, then a few more trees grown on the target's last N months
    and, for reference, the models scored on the SOURCE sites in the new period ("trained_on_sources").

"Last N months" always ends at the cut: the validation start in a development run, test_start in the
final run. So nothing the models learn from is later than anything they are scored on.

Writes <out>/<phase>/per_site.csv (one row per model, setting and site), summary.csv (pooled) and
daily_errors.csv (error sums per site and day, used by report.py to say which differences are real).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from data import load_dataset
from evaluate import daily_errors, evaluate_sites, metrics
from models import FixedFactor, RollingFactor, build_model
from splits import history_window, make_periods, read_roles, slice_period


def run_experiment(tables, sources, targets, cfg, cut, score_end, build=build_model, log=print, daily=None):
    """Train and score everything. Returns (per-site table, pooled summary table).
    tables: {ss_id: site table}; cut: first moment that may NOT be trained on;
    score_end: end of the scored period (None = end of data).
    daily: pass an empty list to also collect the per-day error sums (for the bootstrap in report.py)."""
    cut = pd.Timestamp(cut)
    months_list = list(cfg.get("finetune_months", [1, 3, 12]))
    rolling = f"rolling_{cfg.get('context_hours', 168) // 24}d"
    source_frames = [slice_period(tables[i], None, cut) for i in sources]
    parts, pools, skipped = [], {}, []
    t0 = time.time()

    def score(name, setting, role, model, ids):
        per_site, scored = evaluate_sites(model, tables, ids, cut, score_end, cfg)
        parts.append(per_site.reset_index().assign(model=name, setting=setting, role=role))
        pools.setdefault((name, setting, role), []).append(scored)
        if daily is not None:
            daily.append(daily_errors(scored).assign(model=name, setting=setting, role=role))

    # baseline that learns nothing
    base = build("day_before", cfg)
    score("day_before", "none", "target", base, targets)
    score("day_before", "none", "source", base, sources)

    for name in ("linear_ssrd", "xgb"):
        log(f"[{time.time() - t0:5.0f}s] {name}: training on {len(sources)} source sites")
        source_model = build(name, cfg).fit(source_frames)
        score(name, "trained_on_sources", "source", source_model, sources)
        score(name, "zero_shot", "target", source_model, targets)
        score(name, rolling, "target", RollingFactor(source_model, hours=cfg.get("context_hours", 168)), targets)
        for months in months_list:
            start, _ = history_window(cut, months)
            for t in targets:
                window = [slice_period(tables[t], start, cut)]
                try:
                    score(name, f"calibrated_{months}m", "target", FixedFactor(source_model, window), [t])
                    score(name, f"scratch_{months}m", "target", build(name, cfg).fit(window), [t])
                    if hasattr(source_model, "fine_tune"):
                        tuned = source_model.fine_tune(window, cfg.get("finetune_trees", 50),
                                                       cfg.get("finetune_learning_rate", 0.1))
                        score(name, f"finetune_{months}m", "target", tuned, [t])
                except ValueError as err:          # e.g. no usable row in the window
                    skipped.append((name, t, months, str(err)))
            log(f"[{time.time() - t0:5.0f}s] {name}: {months}-month windows done")
    for name, t, months, why in skipped:
        log(f"  skipped {name} target {t} {months}m: {why}")

    per_site = pd.concat(parts, ignore_index=True)
    return per_site, summarise(per_site, pools, months_list, rolling)


def summarise(per_site: pd.DataFrame, pools: dict, months_list, rolling: str) -> pd.DataFrame:
    """Pooled metrics (all scored hours of all sites of a group together) plus the average of the
    per-site scores, and the skill against the day-before baseline."""
    rows = []
    for (model, setting, role), scored_parts in pools.items():
        sub = per_site[(per_site.model == model) & (per_site.setting == setting) & (per_site.role == role)]
        rows.append({"model": model, "setting": setting, "role": role, "n_sites": sub.ss_id.nunique(),
                     **metrics(pd.concat(scored_parts)), "mean_site_rmse_day": sub.rmse_day.mean()})
    summary = pd.DataFrame(rows)
    reference = summary[summary.model == "day_before"].set_index("role")["rmse_day"]
    summary["skill_vs_day_before"] = 1.0 - summary["rmse_day"] / summary["role"].map(reference)
    order = ["none", "trained_on_sources", "zero_shot", rolling] + [
        f"{kind}_{m}m" for m in months_list for kind in ("calibrated", "scratch", "finetune")]
    summary["setting"] = pd.Categorical(summary["setting"], order, ordered=True)
    return summary.sort_values(["role", "setting", "model"], ascending=[False, True, True]).reset_index(drop=True)


def load_tables(cfg: dict, ids: list[int]) -> dict[int, pd.DataFrame]:
    out_dir = Path(cfg["paths"]["out_dir"])
    tables = {}
    for i in ids:
        path = out_dir / f"site_{i}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing - run data.py first (and check sites.csv)")
        tables[i] = load_dataset(path)[0]
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the transfer experiment.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--final", action="store_true", help="score the TEST period (use once)")
    parser.add_argument("--quick", action="store_true", help="5 sources and 2 targets only")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    sources, targets = read_roles(cfg["sites_csv"])
    if args.quick:
        sources, targets = sources[:5], targets[:2]
    p = make_periods(cfg)
    if args.final:
        phase, cut, score_end = "final", p["test_start"], None
        print("FINAL run: scoring the TEST period. Do this once, after all settings are decided.")
    else:
        phase, cut, score_end = "dev", p["val_start"], p["test_start"]
    print(f"{phase}: train before {cut.date()}, score {cut.date()} to "
          f"{'end of data' if score_end is None else score_end.date()}; "
          f"{len(sources)} sources, {len(targets)} targets")

    tables = load_tables(cfg, sources + targets)
    daily = []
    per_site, summary = run_experiment(tables, sources, targets, cfg, cut, score_end, daily=daily)

    out = Path(args.out) / phase / ("quick" if args.quick else "")
    out.mkdir(parents=True, exist_ok=True)
    per_site.round(5).to_csv(out / "per_site.csv", index=False)
    summary.round(5).to_csv(out / "summary.csv", index=False)
    pd.concat(daily, ignore_index=True).to_csv(out / "daily_errors.csv", index=False)
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    cols = ["role", "setting", "model", "n_sites", "n_blocks", "rmse_day", "mae_day", "bias_day",
            "skill_vs_day_before"]
    print("\nPooled over the sites of each group; errors in kWh per kWp per hour, daytime hours.")
    print("skill_vs_day_before: 0 = no better than yesterday's output, 1 = perfect.\n")
    print(summary[cols].round(4).to_string(index=False))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
