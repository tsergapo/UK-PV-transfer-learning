"""lstm.py - an LSTM forecaster with the same contract as the models in models.py
(fit / predict / fine_tune / context_hours), so run.py and evaluate.py treat it like any other model.

How it forecasts one day (the 24 hours 00:00-23:00), step by step:

    past 168 hours  : [own output, weather, calendar] per hour  --LSTM-->  one summary vector h
    each hour of the day : [that hour's weather and calendar, h]  --small network-->  that hour's output

  * The LSTM reads the site's OWN recent output, so it can pick up a site's scale by itself (what the
    rolling factor does by hand). That is information XGBoost doesn't get, so "zero_shot" for this
    model means: trained on the source sites only, but reading 168 hours of the new site's recent output.
  * All 24 hours are predicted at once from that summary. Nothing is fed back, so errors don't pile up.
  * Training examples are days (windows) cut from the site tables. Windows never mix sites.

Needs PyTorch. A GPU makes training much faster (Colab: Runtime -> Change runtime type -> GPU).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from models import TARGET

DEFAULTS = {
    "hidden": 64, "dropout": 0.1,
    "epochs": 40, "patience": 5, "batch_size": 256, "lr": 2e-3, "weight_decay": 1e-5, "grad_clip": 1.0,
    "stride": 24,              # training windows start every 24 h (one per day) when training on many sites
    "val_fraction": 0.15,      # the last part of each site's windows is used to stop training in time
    "finetune_epochs": 10, "finetune_lr": 3e-4, "finetune_stride": 1,   # fine-tuning on a short window: every hour
    "context": 168, "horizon": 24, "seed": 0,
}


# ----------------------------------------------------------------------------------------------
# Plain numpy helpers (no torch): which windows exist, and which rows they use
# ----------------------------------------------------------------------------------------------
class Scaler:
    """z-score with the mean and spread of the training data (NaN ignored)."""
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    @classmethod
    def fit(cls, values: np.ndarray) -> Scaler:
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        return cls(mean, np.where(std < 1e-8, 1.0, std))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


def window_starts(ok_past, ok_future, hours, context: int, horizon: int, stride: int) -> np.ndarray:
    """Row positions s of usable windows: the `context` rows before s are ok_past, the `horizon` rows
    from s on are ok_future, and the hour of day of s is a multiple of `stride` (24 = midnight only).
    ok_* are boolean arrays over the rows of one site table."""
    n = len(ok_past)
    bad_past = np.concatenate(([0], np.cumsum(~ok_past)))
    bad_future = np.concatenate(([0], np.cumsum(~ok_future)))
    s = np.flatnonzero(np.asarray(hours) % stride == 0)
    s = s[(s >= context) & (s + horizon <= n)]
    clean = (bad_past[s] - bad_past[s - context] == 0) & (bad_future[s + horizon] - bad_future[s] == 0)
    return s[clean]


def window_rows(starts: np.ndarray, context: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Row numbers of each window: (past rows [B, context], future rows [B, horizon])."""
    starts = np.asarray(starts)
    return (starts[:, None] + np.arange(-context, 0)[None, :],
            starts[:, None] + np.arange(horizon)[None, :])


# ----------------------------------------------------------------------------------------------
# The network
# ----------------------------------------------------------------------------------------------
class Net(nn.Module):
    """past [B, context, 1 + F] and future [B, horizon, F]  ->  forecast [B, horizon] (standardised)."""
    def __init__(self, n_inputs: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.LSTM(1 + n_inputs, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(n_inputs + hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, past, future):
        _, (h, _) = self.encoder(past)                                  # h: [1, B, hidden]
        h = h[-1][:, None, :].expand(-1, future.shape[1], -1)           # the same summary for every hour
        return self.head(torch.cat([future, h], dim=-1)).squeeze(-1)


# ----------------------------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------------------------
class LSTMModel:
    name = "lstm"

    def __init__(self, inputs: list[str], target: str = TARGET, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown lstm settings: {sorted(unknown)}")
        if target in inputs:
            raise ValueError("the target must not be one of the inputs")
        self.inputs, self.target, self.p = list(inputs), target, {**DEFAULTS, **params}
        self.context_hours = self.p["context"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.y_scaler = self.x_scaler = None

    # ---- data ----
    def _arrays(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        y = self.y_scaler.transform(df[[self.target]].to_numpy(dtype="float64"))[:, 0]
        x = self.x_scaler.transform(df[self.inputs].to_numpy(dtype="float64"))
        return y.astype("float32"), x.astype("float32")

    def _collect(self, frames, stride: int):
        """All site tables as one long array, and the usable window starts of each site (global rows)."""
        ys, xs, starts, offset = [], [], [], 0
        for df in frames:
            y, x = self._arrays(df)
            ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
            s = window_starts(ok, ok, df.index.hour.to_numpy(), self.p["context"], self.p["horizon"], stride)
            ys.append(y), xs.append(x), starts.append(s + offset)
            offset += len(df)
        if sum(len(s) for s in starts) == 0:
            raise ValueError("no clean window to learn from (too many gaps, or too little data)")
        return np.concatenate(ys), np.concatenate(xs), starts

    def _batch(self, Y, X, starts):
        past_rows, future_rows = window_rows(starts, self.p["context"], self.p["horizon"])
        pr = torch.as_tensor(past_rows, device=Y.device)
        fr = torch.as_tensor(future_rows, device=Y.device)
        past = torch.cat([Y[pr].unsqueeze(-1), X[pr]], dim=-1)           # own output + covariates, past hours
        return past, X[fr], Y[fr]                                        # future covariates, true output

    # ---- training ----
    def _train(self, net, Y, X, starts, epochs: int, lr: float, use_val: bool) -> None:
        p = self.p
        train, val = [], []
        for s in starts:
            cut = int(len(s) * (1 - p["val_fraction"])) if use_val else len(s)   # time-ordered: latest windows validate
            train.append(s[:cut]), val.append(s[cut:])
        train, val = np.concatenate(train), np.concatenate(val)
        if use_val and len(train) + len(val) < 200:                       # too few windows to hold any back
            train, val = np.concatenate([train, val]), val[:0]
        verbose = len(train) > 5000                                       # only the big training prints progress
        Yt = torch.as_tensor(Y, device=self.device)
        Xt = torch.as_tensor(X, device=self.device)
        gen = torch.Generator()
        gen.manual_seed(p["seed"])
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=p["weight_decay"])
        loss_fn = nn.MSELoss()
        best, best_state, bad, skipped = np.inf, None, 0, 0

        for epoch in range(1, epochs + 1):
            net.train()
            order = train[torch.randperm(len(train), generator=gen).numpy()]
            total, count = 0.0, 0
            for i in range(0, len(order), p["batch_size"]):
                past, future, truth = self._batch(Yt, Xt, order[i:i + p["batch_size"]])
                opt.zero_grad()
                loss = loss_fn(net(past, future), truth)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"lstm: non-finite training loss in epoch {epoch} "
                                             f"(inputs finite: {bool(torch.isfinite(past).all())})")
                loss.backward()
                if not np.isfinite(float(nn.utils.clip_grad_norm_(net.parameters(), p["grad_clip"]))):
                    skipped += 1                                          # skip a step with NaN gradients
                    if skipped > 50:
                        raise FloatingPointError("lstm: too many steps with non-finite gradients")
                    continue
                opt.step()
                total += loss.item() * len(truth)
                count += len(truth)
            if len(val):
                net.eval()
                with torch.no_grad():
                    errs = [loss_fn(net(*self._batch(Yt, Xt, val[j:j + 1024])[:2]),
                                    self._batch(Yt, Xt, val[j:j + 1024])[2]).item() * len(val[j:j + 1024])
                            for j in range(0, len(val), 1024)]
                val_loss = sum(errs) / len(val)
                if verbose:
                    print(f"[lstm] epoch {epoch:3d}  train {total / max(count, 1):.4f}  validation {val_loss:.4f}", flush=True)
                if val_loss < best - 1e-9:
                    best, bad = val_loss, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= p["patience"]:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)                               # go back to the best epoch
        net.eval()

    def fit(self, frames: list[pd.DataFrame]) -> LSTMModel:
        columns = [self.target, *self.inputs]
        pooled = np.concatenate([f[columns].to_numpy(dtype="float64") for f in frames])
        self.y_scaler, self.x_scaler = Scaler.fit(pooled[:, :1]), Scaler.fit(pooled[:, 1:])
        Y, X, starts = self._collect(frames, self.p["stride"])
        torch.manual_seed(self.p["seed"])
        self.net = Net(len(self.inputs), self.p["hidden"], self.p["dropout"]).to(self.device)
        self._train(self.net, Y, X, starts, self.p["epochs"], self.p["lr"], use_val=True)
        return self

    def fine_tune(self, frames: list[pd.DataFrame], epochs: int | None = None, lr: float | None = None) -> LSTMModel:
        """A NEW model: this one trained a little more on `frames` (e.g. a target site's last months),
        with a small learning rate and a fixed number of epochs. This model stays unchanged."""
        if self.net is None:
            raise RuntimeError("fit the model before fine-tuning it")
        Y, X, starts = self._collect(frames, self.p["finetune_stride"])   # same scalers as the source model
        tuned = copy.copy(self)
        tuned.net = copy.deepcopy(self.net)
        self._train(tuned.net, Y, X, starts, epochs or self.p["finetune_epochs"],
                    lr or self.p["finetune_lr"], use_val=False)
        return tuned

    # ---- forecasting ----
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """For every midnight with 168 clean past hours: the 24 hours of that day. NaN elsewhere."""
        p = self.p
        y, x = self._arrays(df)
        ok_past = np.isfinite(y) & np.isfinite(x).all(axis=1)
        ok_future = np.isfinite(x).all(axis=1)                            # the day's own output is not needed
        starts = window_starts(ok_past, ok_future, df.index.hour.to_numpy(), p["context"], p["horizon"], 24)
        out = np.full(len(df), np.nan)
        if len(starts):
            Yt, Xt = torch.as_tensor(y, device=self.device), torch.as_tensor(x, device=self.device)
            parts = []
            with torch.no_grad():
                for j in range(0, len(starts), 512):
                    past, future, _ = self._batch(Yt, Xt, starts[j:j + 512])
                    parts.append(self.net(past, future).cpu().numpy())
            forecast = np.clip(self.y_scaler.inverse(np.concatenate(parts)), 0.0, None)
            out[starts[:, None] + np.arange(p["horizon"])[None, :]] = forecast
        return pd.Series(out, index=df.index, name=self.name)
