"""Tests for data.py. They use tiny made-up data, so they run in seconds without any real files.
Run with:  pytest"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data

LAT, LON = 51.5, -1.8


def make_raw(values=None, start="2021-01-10", days=5, drop=()):
    """Half-hourly raw readings of zeros; `values` sets chosen stamps. Stamps are period ENDS."""
    stamps = pd.date_range(start, periods=48 * days, freq="30min") + pd.Timedelta(minutes=30)
    raw = pd.DataFrame({"datetime_GMT": stamps, "generation_Wh": np.zeros(len(stamps))})
    for stamp, value in (values or {}).items():
        raw.loc[raw.datetime_GMT == pd.Timestamp(stamp), "generation_Wh"] = value
    return raw[~raw.datetime_GMT.isin([pd.Timestamp(d) for d in drop])]


def test_sun_is_up_at_noon_and_down_at_midnight_in_summer():
    idx = pd.DatetimeIndex(["2021-06-21 12:00", "2021-06-21 00:00"])
    noon, midnight = data.solar_elevation(idx, LAT, LON)
    assert noon > 55 and midnight < 0


def test_a_stamp_marks_the_end_of_its_half_hour():
    # stamps 12:30 and 13:00 are the half-hours 12:00-12:30 and 12:30-13:00 -> the 12:00 hour
    raw = make_raw({"2021-01-12 12:30": 1000, "2021-01-12 13:00": 500})
    hourly = data.clean_pv(raw, LAT, LON)
    assert abs(hourly.loc["2021-01-12 12:00"] - 1.5) < 1e-9      # (1000 + 500) Wh = 1.5 kWh
    assert hourly.loc["2021-01-12 11:00"] == 0 and hourly.loc["2021-01-12 13:00"] == 0


def test_one_missing_daytime_reading_is_interpolated_but_a_long_gap_is_not():
    day = {f"2021-01-12 {h:02d}:{m:02d}": 100 for h in range(9, 15) for m in (0, 30)}
    one = data.clean_pv(make_raw(day, drop=["2021-01-12 11:00"]), LAT, LON)
    assert one.loc["2021-01-12 10:00":"2021-01-12 11:00"].notna().all()
    long = data.clean_pv(make_raw(day, drop=["2021-01-12 11:00", "2021-01-12 11:30",
                                             "2021-01-12 12:00"]), LAT, LON)
    assert long.loc["2021-01-12 11:00":"2021-01-12 12:00"].isna().any()


def test_missing_night_readings_become_zero_and_the_timeline_has_no_holes():
    hourly = data.clean_pv(make_raw(drop=["2021-01-12 02:00", "2021-01-12 02:30"]), LAT, LON)
    assert hourly.loc["2021-01-12 01:00":"2021-01-12 02:00"].notna().all()
    assert (hourly.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_two_stamps_that_round_to_the_same_time_do_not_crash():
    raw = make_raw({"2021-01-12 12:30": 1000})
    extra = pd.DataFrame({"datetime_GMT": [pd.Timestamp("2021-01-12 12:30:10")],
                          "generation_Wh": [5.0]})
    hourly = data.clean_pv(pd.concat([raw, extra], ignore_index=True), LAT, LON)
    assert hourly.index.is_unique and abs(hourly.loc["2021-01-12 12:00"] - 1.0) < 1e-9


def test_radiation_for_an_hour_comes_from_the_value_stamped_one_hour_later():
    idx = pd.date_range("2021-06-01 09:00", periods=4, freq="1h")
    weather = pd.DataFrame({"ssrd": [0.0, 1_800_000.0, 3_600_000.0, 7_200_000.0],
                            "t2m": [280.0, 282.0, 284.0, 286.0]}, index=idx)
    aligned = data.align_weather(weather)
    assert aligned.loc["2021-06-01 09:00", "ssrd"] == 500.0   # 1,800,000 J / 3600 s
    assert aligned.loc["2021-06-01 10:00", "ssrd"] == 1000.0
    assert aligned.loc["2021-06-01 09:00", "t2m"] == 281.0    # average of 09:00 and 10:00


def test_small_negative_radiation_is_set_to_zero():
    idx = pd.date_range("2021-06-01 00:00", periods=3, freq="1h")
    weather = pd.DataFrame({"ssrd": [0.0, -5.0, 0.0]}, index=idx)
    assert (data.align_weather(weather)["ssrd"].dropna() >= 0).all()


def _write_era5(tmp_path, name, times, **columns):
    """A CSV with two grid points: one near (51.5, -1.75) and one far away (60, -1.75)."""
    near = pd.DataFrame({"latitude": 51.5, "longitude": -1.75, "valid_time": times, **columns})
    far = near.assign(latitude=60.0)
    pd.concat([near, far]).to_csv(tmp_path / name, index=False)
    return tmp_path / name


def test_era5_files_with_different_variables_or_periods_are_combined(tmp_path):
    t1 = pd.date_range("2020-01-01", periods=3, freq="1h")
    t2 = pd.date_range("2020-01-01 03:00", periods=3, freq="1h")
    files = [_write_era5(tmp_path, "a1.csv", t1, ssrd=1.0, fdir=2.0),   # same variables, period 1
             _write_era5(tmp_path, "a2.csv", t2, ssrd=3.0, fdir=4.0),   # same variables, period 2
             _write_era5(tmp_path, "b.csv", t1.append(t2), t2m=280.0)]  # other variables, all hours
    weather = data.weather_at(data.read_era5_csv(files), lat=51.6, lon=-1.8)
    assert len(weather) == 6 and set(weather.columns) == {"ssrd", "fdir", "t2m"}
    assert weather.notna().all().all()
    assert weather["ssrd"].tolist() == [1.0, 1.0, 1.0, 3.0, 3.0, 3.0]   # the near point, not 60N


def test_each_site_gets_its_own_nearest_grid_point(tmp_path):
    t = pd.date_range("2020-01-01", periods=2, freq="1h")
    south = pd.DataFrame({"latitude": 51.25, "longitude": -2.0, "valid_time": t, "ssrd": 1.0})
    north = south.assign(latitude=51.75, ssrd=2.0)
    path = tmp_path / "w.csv"
    pd.concat([south, north]).to_csv(path, index=False)
    points = data.read_era5_csv([path])
    assert data.weather_at(points, 51.3, -2.0)["ssrd"].iloc[0] == 1.0
    assert data.weather_at(points, 51.7, -2.0)["ssrd"].iloc[0] == 2.0


def test_yield_is_output_divided_by_size_and_orientation_is_a_circle():
    idx = pd.date_range("2021-06-01", periods=3, freq="1h")
    df = pd.DataFrame({"generation_kwh": [1.0, 2.0, 4.0]}, index=idx)
    out = data.add_site_columns(df, {"kwp": 2.0, "tilt": 35.0, "orientation": 90.0})
    assert out["yield_kwh_per_kwp"].tolist() == [0.5, 1.0, 2.0]
    assert out["tilt"].iloc[0] == 35.0
    assert abs(out["orient_sin"].iloc[0] - 1.0) < 1e-9 and abs(out["orient_cos"].iloc[0]) < 1e-9


def test_month_coverage_counts_complete_days_and_ignores_nights():
    idx = pd.date_range("2021-01-01", "2021-03-31 23:00", freq="1h")
    day = ((idx.hour >= 9) & (idx.hour < 15)).astype(float)       # a fake daylight flag
    df = pd.DataFrame({"generation_kwh": np.where(day > 0, 1.0, np.nan), "daylight": day}, index=idx)
    df.loc["2021-02-10 11:00", "generation_kwh"] = np.nan            # one daytime hour missing
    days = data.month_coverage(df)
    assert days.loc[1] == 31 and days.loc[2] == 27 and days.loc[3] == 31 and days.loc[4] == 0
    assert data.month_coverage(df, before="2021-02-01").loc[2] == 0   # only earlier rows count


def test_build_all_makes_one_table_per_site_from_raw_files(tmp_path, monkeypatch):
    # raw PV for two sites (parquet reading is replaced by in-memory tables), metadata, weather, site list
    stamps = pd.date_range("2021-06-01", periods=48 * 4, freq="30min") + pd.Timedelta(minutes=30)
    raw = pd.concat([pd.DataFrame({"ss_id": i, "datetime_GMT": stamps, "generation_Wh": 100.0 * i})
                     for i in (1, 2)])
    monkeypatch.setattr(data, "read_pv", lambda pv_dir, ids: {i: raw[raw.ss_id == i] for i in ids})
    pd.DataFrame({"ss_id": [1, 2, 3], "latitude_rounded": [51.5, 51.5, 51.5],
                  "longitude_rounded": [-1.75] * 3, "kWp": [2.0, 4.0, 3.0], "tilt": [30, 40, 35],
                  "orientation": [180, 90, 180]}).to_csv(tmp_path / "meta.csv", index=False)
    pd.DataFrame({"ss_id": [1, 2], "role": ["source", "target"]}).to_csv(tmp_path / "sites.csv", index=False)
    hours = pd.date_range("2021-05-31", "2021-06-06", freq="1h")
    w = {"ssrd": 3.6e6, "fdir": 1.8e6, "tcc": 0.5, "t2m": 288.0, "u10": 3.0, "v10": 4.0}
    _write_era5(tmp_path, "w.csv", hours, **w)
    cfg = {"sites_csv": str(tmp_path / "sites.csv"), "max_gap_halfhours": 1,
           "paths": {"pv_dir": "unused", "metadata_csv": str(tmp_path / "meta.csv"),
                     "era5_csv": [str(tmp_path / "w.csv")]}}
    result = data.build_all(cfg, log=lambda m: None)
    assert set(result) == {1, 2}
    a, site_a = result[1]
    b, _ = result[2]
    assert site_a["tilt"] == 30.0 and {"yield_kwh_per_kwp", "orient_sin", "ssrd"} <= set(a.columns)
    noon = "2021-06-02 12:00"
    # site 1: 100 Wh per half-hour / 2 kWp, site 2: 200 Wh per half-hour / 4 kWp -> same yield
    assert abs(a.loc[noon, "yield_kwh_per_kwp"] - b.loc[noon, "yield_kwh_per_kwp"]) < 1e-9
    assert abs(a.loc[noon, "generation_kwh"] - 0.2) < 1e-9


def test_assembled_table_keeps_gaps_as_nan_and_has_the_feature_columns():
    idx = pd.date_range("2021-06-01", periods=24 * 6, freq="1h")
    weather = pd.DataFrame({"ssrd": 3.6e6, "fdir": 1.8e6, "tcc": 0.5, "t2m": 288.0,
                            "u10": 3.0, "v10": 4.0}, index=idx)
    pv = pd.Series(1.0, index=idx[5:-5], name="generation_kwh")
    pv.loc["2021-06-03 10:00":"2021-06-03 12:00"] = np.nan
    table = data.assemble(pv, weather, LAT, LON)
    assert table.loc["2021-06-03 11:00", "generation_kwh"] != table.loc["2021-06-03 11:00", "generation_kwh"]
    assert (table.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    assert table["ws10"].iloc[0] == 5.0 and abs(table["t2m_c"].iloc[0] - 14.85) < 1e-9
    assert {"hour_sin", "doy_cos", "solar_elevation", "daylight"} <= set(table.columns)
