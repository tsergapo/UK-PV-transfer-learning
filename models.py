"""models.py - the forecasting models. Every model has the same three things:

    model.fit(frames)       learn from a list of site tables (as built by data.py)
    model.predict(table)    one forecast per row of ONE site table (NaN where it cannot forecast)
    model.context_hours     how many hours of the site's own past output it needs before a forecast
                            (0 = none; evaluation only scores blocks where that much past exists)

The target everywhere is `yield_kwh_per_kwp` (hourly output per installed kWp).

  XGBModel       gradient-boosted trees on same-hour weather, calendar, sun and site columns.
                 Rows are independent, so training on many sites is just stacking their rows.
                 `fine_tune` continues an already trained model with a few more trees.
  DayBefore      baseline: the output of the same hour yesterday.
  LinearRadiation baseline: output = one number x solar radiation (ssrd), the number fitted on data.

If a model cannot beat these two, it has learned nothing useful.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

TARGET = "yield_kwh_per_kwp"

# the same settings the original notebook used (xgboost's own defaults, written out)
XGB_DEFAULTS = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3,
                "tree_method": "hist", "random_state": 0, "n_jobs": -1}


def stack(frames: list[pd.DataFrame], columns: list[str], target: str = TARGET):
    """Inputs X and target y from many site tables: all their rows one under the other,
    keeping only the rows where every needed value exists."""
    missing = [c for c in [*columns, target] if c not in frames[0].columns]
    if missing:
        raise KeyError(f"columns not in the tables: {missing}")
    big = pd.concat([f[[*columns, target]] for f in frames])
    values = big.to_numpy(dtype="float32")
    ok = np.isfinite(values).all(axis=1)
    if not ok.any():
        raise ValueError("no row has all inputs and the target - nothing to learn from")
    return values[ok][:, :-1], values[ok][:, -1]


class XGBModel:
    name = "xgb"
    context_hours = 0

    def __init__(self, inputs: list[str], target: str = TARGET, estimator_factory=None, **params):
        if target in inputs:
            raise ValueError("the target must not be one of the inputs")
        self.inputs, self.target = list(inputs), target
        self.params = {**XGB_DEFAULTS, **params}
        self._factory = estimator_factory or self._xgboost   # a stand-in can be passed for tests
        self.est = None

    def _xgboost(self):
        from xgboost import XGBRegressor
        return XGBRegressor(**self.params)

    def fit(self, frames: list[pd.DataFrame]) -> XGBModel:
        X, y = stack(frames, self.inputs, self.target)
        self.est = self._factory()
        self.est.fit(X, y)
        return self

    def fine_tune(self, frames: list[pd.DataFrame], n_trees: int = 50, learning_rate: float = 0.1):
        """A NEW model: this one plus `n_trees` more trees grown on `frames` (e.g. a target
        site's last months). This model is left unchanged, so it can be fine-tuned again."""
        if self.est is None:
            raise RuntimeError("fit the model before fine-tuning it")
        X, y = stack(frames, self.inputs, self.target)
        est = copy.deepcopy(self.est)
        est.set_params(n_estimators=n_trees, learning_rate=learning_rate)
        est.fit(X, y, xgb_model=self.est.get_booster())      # keep the old trees, add new ones
        tuned = copy.copy(self)
        tuned.est = est
        return tuned

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.inputs].to_numpy(dtype="float32")
        ok = np.isfinite(X).all(axis=1)
        out = np.full(len(df), np.nan)
        if ok.any():
            out[ok] = np.clip(np.asarray(self.est.predict(X[ok]), dtype=float), 0.0, None)
        return pd.Series(out, index=df.index, name=self.name)


class DayBefore:
    """Baseline: forecast each hour with the output of the same hour 24 hours earlier."""
    name = "day_before"
    context_hours = 24

    def __init__(self, target: str = TARGET):
        self.target = target

    def fit(self, frames):
        return self                                          # nothing to learn

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if not (df.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
            raise ValueError("DayBefore needs a continuous hourly table")
        return df[self.target].shift(24).rename(self.name)


class LinearRadiation:
    """Baseline: output = slope x ssrd, with the one slope fitted by least squares."""
    name = "linear_ssrd"
    context_hours = 0

    def __init__(self, target: str = TARGET):
        self.target, self.slope = target, None

    def fit(self, frames):
        x, y = stack(frames, ["ssrd"], self.target)
        x = x[:, 0].astype("float64")
        self.slope = float((x * y).sum() / (x * x).sum())
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return (self.slope * df["ssrd"]).clip(lower=0.0).rename(self.name)


def build_model(name: str, cfg: dict):
    """A fresh, untrained model by name: 'xgb', 'day_before' or 'linear_ssrd'."""
    if name == "xgb":
        return XGBModel(cfg["inputs"], **cfg.get("xgb", {}))
    if name == "day_before":
        return DayBefore()
    if name == "linear_ssrd":
        return LinearRadiation()
    raise ValueError(f"unknown model '{name}' (choose xgb, day_before or linear_ssrd)")
