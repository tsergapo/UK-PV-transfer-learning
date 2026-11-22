"""splits.py - who is trained on, who is tested, and which time periods are used.

Run it:    python splits.py          (reads config.yaml, needs data/processed from data.py)
It prints the site roles, the time periods, and how much usable history each target has.

The whole design is one time boundary, `test_start` (config.yaml):

        sources:   |---------- train ----------|-- validation --|----- test -----|
        targets:   |   (own history: the last N months before the boundary)  |----- test -----|
                                                           test_start

  * SOURCE sites are used for training, only with rows BEFORE test_start. After the boundary they
    are scored too (same sites, new years).
  * TARGET sites are never trained on, except through their own history just before test_start
    (the last 1, 3 or 12 months) for fine-tuning or training from scratch. After the boundary
    they are scored.
  * Because everything trained on lies before the boundary and everything scored lies after it,
    no model has seen the weather or the output of any scored hour - not even through another
    site (nearby sites share weather, which is why the boundary is common to all sites).

All functions are plain: they take values or tables and return values or tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from data import complete_days, load_dataset


def read_roles(sites_csv: str | Path) -> tuple[list[int], list[int]]:
    """(source ids, target ids) from the site list."""
    sites = pd.read_csv(sites_csv)
    if not {"ss_id", "role"} <= set(sites.columns):
        raise ValueError(f"{sites_csv} needs the columns ss_id and role")
    unknown = set(sites["role"]) - {"source", "target"}
    if unknown:
        raise ValueError(f"role must be 'source' or 'target', found {sorted(unknown)}")
    sources = sites.loc[sites["role"] == "source", "ss_id"].astype(int).tolist()
    targets = sites.loc[sites["role"] == "target", "ss_id"].astype(int).tolist()
    if set(sources) & set(targets):
        raise ValueError("a site cannot be both source and target")
    return sources, targets


def make_periods(cfg: dict) -> dict[str, pd.Timestamp]:
    """The two dates that cut time: where validation starts and where the test period starts."""
    test_start = pd.Timestamp(cfg["test_start"])
    val_start = test_start - pd.DateOffset(months=cfg.get("val_months", 12))
    return {"val_start": val_start, "test_start": test_start}


def slice_period(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Rows from `start` up to but NOT including `end` (None = no limit on that side).
    Half-open on purpose: the period before a boundary and the period after it never share a row."""
    keep = pd.Series(True, index=df.index)
    if start is not None:
        keep &= df.index >= pd.Timestamp(start)
    if end is not None:
        keep &= df.index < pd.Timestamp(end)
    return df.loc[keep]


def history_window(test_start, months: int | None) -> tuple[pd.Timestamp | None, pd.Timestamp]:
    """The slice of a target's own past that it may use: the last `months` before the boundary.
    months=None means all of its history (a reference, not a realistic starting point)."""
    test_start = pd.Timestamp(test_start)
    start = None if months is None else test_start - pd.DateOffset(months=months)
    return start, test_start


def window_report(tables: dict[int, pd.DataFrame], ids: list[int], test_start, months_list) -> pd.DataFrame:
    """Complete days each site really has in each history window. Calendar length is not
    enough: a window can contain gaps (the dataset has some in 2021 and 2022)."""
    rows = {}
    for i in ids:
        row = {}
        for m in months_list:
            start, end = history_window(test_start, m)
            row["all" if m is None else f"{m}m"] = int(complete_days(slice_period(tables[i], start, end)).sum())
        rows[i] = row
    return pd.DataFrame(rows).T.rename_axis("ss_id")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the roles, periods and target history windows.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    sources, targets = read_roles(cfg["sites_csv"])
    p = make_periods(cfg)
    print(f"{len(sources)} source sites, {len(targets)} target sites")
    print(f"train:      before {p['val_start'].date()}   (final run: before {p['test_start'].date()})")
    print(f"validation: {p['val_start'].date()} to {p['test_start'].date()}")
    print(f"test:       from {p['test_start'].date()} on\n")

    tables = {i: load_dataset(Path(cfg["paths"]["out_dir"]) / f"site_{i}.parquet")[0] for i in targets}
    months = list(cfg.get("finetune_months", [1, 3, 12])) + [None]
    print("complete days in each target's history window (calendar length: 1m=~30, 3m=~90, 12m=~365):")
    print(window_report(tables, targets, p["test_start"], months).to_string())


if __name__ == "__main__":
    main()
