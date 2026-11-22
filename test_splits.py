"""Tests for splits.py (tiny made-up tables, no real files)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splits


def table(start="2022-01-01", end="2023-12-31 23:00"):
    idx = pd.date_range(start, end, freq="1h")
    day = ((idx.hour >= 9) & (idx.hour < 15)).astype(float)
    return pd.DataFrame({"generation_kwh": np.where(day > 0, 1.0, 0.0), "daylight": day}, index=idx)


def test_roles_are_read_and_checked(tmp_path):
    path = tmp_path / "s.csv"
    pd.DataFrame({"ss_id": [1, 2, 3], "role": ["source", "target", "source"]}).to_csv(path, index=False)
    assert splits.read_roles(path) == ([1, 3], [2])
    pd.DataFrame({"ss_id": [1], "role": ["train"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        splits.read_roles(path)
    pd.DataFrame({"ss_id": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        splits.read_roles(path)


def test_periods_come_from_the_boundary_and_validation_length():
    p = splits.make_periods({"test_start": "2023-04-02", "val_months": 12})
    assert p["test_start"] == pd.Timestamp("2023-04-02") and p["val_start"] == pd.Timestamp("2022-04-02")


def test_the_period_before_a_boundary_and_the_period_after_never_share_a_row():
    df = table()
    boundary = pd.Timestamp("2023-04-02")
    before, after = splits.slice_period(df, None, boundary), splits.slice_period(df, boundary, None)
    assert before.index.max() == boundary - pd.Timedelta(hours=1) and after.index.min() == boundary
    assert len(before) + len(after) == len(df) and before.index.intersection(after.index).empty


def test_history_window_is_the_last_months_before_the_boundary():
    start, end = splits.history_window("2023-04-02", 3)
    assert start == pd.Timestamp("2023-01-02") and end == pd.Timestamp("2023-04-02")
    assert splits.history_window("2023-04-02", None)[0] is None
    w = splits.slice_period(table(), start, end)
    assert w.index.min() == start and w.index.max() == end - pd.Timedelta(hours=1)   # nothing from the test period


def test_window_report_counts_complete_days_not_calendar_days():
    df = table()
    df.loc["2023-03-10":"2023-03-14", "generation_kwh"] = np.nan        # 5 daytime days lost
    rep = splits.window_report({7: df}, [7], "2023-04-02", [1, 3, None])
    assert rep.loc[7, "1m"] == 31 - 5 and rep.loc[7, "3m"] == 90 - 5
    assert rep.loc[7, "all"] == (pd.Timestamp("2023-04-02") - pd.Timestamp("2022-01-01")).days - 5
