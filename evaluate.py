"""evaluate.py - score a model on one-day forecasts, the same way for every model.

A forecast is a BLOCK: the 24 hours from 00:00 UTC to 23:00 of one day. Every model is scored on
exactly the same blocks of a site, so differences in the scores come from the models only.

A block is used only if
  * it lies inside the period being scored (e.g. the test period), and
  * the site has `context_hours` of clean data before it and the block itself is clean
    (target and all inputs present). This holds even for models that need no past at all, so
    the simple model and the neural networks (which need 168 h of past) see identical blocks.

Main entry point:   per_site, scored = evaluate_sites(model, tables, ids, start, end, cfg)
  per_site : one row of metrics per site            scored : every scored hour (y, pred, ...)
Pooled metrics over several sites:                  metrics(scored)

Metrics are in kWh per kWp per hour (the target's unit), so sites of different size compare.
"daytime" = hours with the sun above the horizon; night hours are almost all zeros and make
every score look better than it is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models import TARGET


def block_starts(df: pd.DataFrame, columns: list[str], start, end, horizon: int = 24,
                 context: int = 168) -> np.ndarray:
    """Row positions of the blocks (each starting at 00:00) inside [start, end) that are usable.
    `columns` are the values that must be present: the target and the model inputs."""
    if not (df.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("the table must be continuous hourly")
    ok = np.isfinite(df[[TARGET, *columns]].to_numpy(dtype="float64")).all(axis=1)
    bad_so_far = np.concatenate(([0], np.cumsum(~ok)))        # bad_so_far[k] = bad rows before row k
    lo = int(df.index.searchsorted(pd.Timestamp(start))) if start is not None else 0
    hi = int(df.index.searchsorted(pd.Timestamp(end))) if end is not None else len(df)
    midnights = np.flatnonzero(df.index.hour == 0)
    pos = midnights[(midnights >= lo) & (midnights + horizon <= hi)]
    first, last = pos - context, pos + horizon                # rows needed: [first, last)
    clean = (first >= 0) & (bad_so_far[last] - bad_so_far[np.maximum(first, 0)] == 0)
    return pos[clean]


def score_blocks(model, df: pd.DataFrame, starts: np.ndarray, horizon: int = 24) -> pd.DataFrame:
    """Forecast every block; one row per scored hour: time, block, y (truth), pred, daylight."""
    pred = model.predict(df).to_numpy(dtype="float64")
    pos = (starts[:, None] + np.arange(horizon)[None, :]).ravel()
    p = pred[pos]
    if not np.isfinite(p).all():
        raise RuntimeError(f"{model.name}: {int((~np.isfinite(p)).sum())} of {len(p)} forecast values "
                           "are NaN/inf in blocks that should be valid")
    return pd.DataFrame({
        "time": df.index[pos],
        "block": np.repeat(df.index[starts].to_numpy(), horizon),
        "y": df[TARGET].to_numpy()[pos],
        "pred": p,
        "daylight": df["daylight"].to_numpy()[pos] > 0.5,
    })


def metrics(scored: pd.DataFrame) -> dict:
    """Errors over all scored hours and over the daytime hours only.
    bias = mean(pred - y): above 0 means the model forecasts too much."""
    out = {"n_blocks": int(len(scored) // 24) if len(scored) else 0, "n_hours": len(scored)}
    for label, part in [("", scored), ("_day", scored[scored["daylight"]])]:
        e = (part["pred"] - part["y"]).to_numpy()
        out[f"rmse{label}"] = float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan
        out[f"mae{label}"] = float(np.mean(np.abs(e))) if len(e) else np.nan
        out[f"bias{label}"] = float(np.mean(e)) if len(e) else np.nan
    return out


def evaluate_sites(model, tables: dict[int, pd.DataFrame], ids: list[int], start, end, cfg: dict):
    """Score `model` on the usable blocks of each site in [start, end).
    Returns (per-site table of metrics, all scored hours of all sites with an ss_id column)."""
    horizon, context = cfg.get("horizon_hours", 24), cfg.get("context_hours", 168)
    if model.context_hours > context:
        raise ValueError(f"{model.name} needs {model.context_hours} h of past, more than the "
                         f"{context} h that every model is required to have")
    rows, parts = [], []
    for i in ids:
        starts = block_starts(tables[i], cfg["inputs"], start, end, horizon, context)
        scored = score_blocks(model, tables[i], starts, horizon)
        rows.append({"ss_id": i, **metrics(scored)})
        parts.append(scored.assign(ss_id=i))
    return pd.DataFrame(rows).set_index("ss_id"), pd.concat(parts, ignore_index=True)


def skill_score(rmse: float, rmse_reference: float) -> float:
    """1 - rmse / rmse_reference: 0 = no better than the reference, 1 = perfect, below 0 = worse."""
    return 1.0 - rmse / rmse_reference
