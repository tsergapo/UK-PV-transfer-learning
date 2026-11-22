"""data.py - from raw files to one clean hourly table PER SITE.

Run it:    python data.py                 (reads config.yaml and the site list it points to)
Result:    data/processed/site_<id>.parquet   (+ site_<id>.json with location, size, tilt, orientation)
           data/processed/month_coverage.csv  (which months of the year each site has data for)

What happens, in order
  1. read each site's location, size (kWp), tilt and orientation from the metadata file
  2. read the half-hourly PV readings and turn them into clean hourly kWh
  3. read the ERA5 weather files ONCE and keep, for each site, the nearest grid point
  4. line the weather up with the PV hours
  5. join PV and weather on one continuous hourly timeline and add calendar / sun / site columns
  6. divide the output by the site's kWp, so a 2 kWp and a 5 kWp roof become comparable

Every function takes plain values and returns plain values. Only `build_all` and `main`
touch files, so each step can be tried on its own in a notebook.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Column names in the raw PV files
TIME_COL = "datetime_GMT"
VALUE_COL = "generation_Wh"

# ERA5 variables that are ACCUMULATED over the hour ending at the timestamp (J/m2)
ACCUMULATED = ("ssrd", "fdir")


# ----------------------------------------------------------------------------------------------
# Sun position (pure geometry, no data needed)
# ----------------------------------------------------------------------------------------------
def solar_elevation(index: pd.DatetimeIndex, lat: float, lon: float, offset_minutes: float = 0.0):
    """Height of the sun above the horizon, in degrees, for UTC timestamps (NOAA approximation).

    offset_minutes moves the moment we evaluate, e.g. 30 = the middle of an hour that starts
    at the timestamp. Above 0 = sun is up, below 0 = night."""
    idx = pd.DatetimeIndex(index) + pd.Timedelta(minutes=offset_minutes)
    doy = idx.dayofyear.to_numpy().astype(float)
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0 + idx.second.to_numpy() / 3600.0

    g = 2.0 * np.pi / 365.0 * (doy - 1.0 + (hour - 12.0) / 24.0)  # fraction of the year (radians)
    eqtime = 229.18 * (  # equation of time: how far the sun runs ahead/behind the clock (minutes)
        0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g)
    )
    decl = (  # declination: how far north/south the sun is on that day (radians)
        0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g)
    )
    solar_minutes = hour * 60.0 + eqtime + 4.0 * lon  # every degree of longitude = 4 minutes
    hour_angle = np.deg2rad(solar_minutes / 4.0 - 180.0)
    lat_r = np.deg2rad(lat)
    cos_zenith = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(hour_angle)
    return 90.0 - np.degrees(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))


# ----------------------------------------------------------------------------------------------
# 1. Site information
# ----------------------------------------------------------------------------------------------
def load_site_info(metadata_csv: str | Path, ss_id: int) -> dict:
    """Location, size, tilt and orientation of one site from metadata.csv."""
    meta = pd.read_csv(metadata_csv)
    row = meta.loc[meta["ss_id"] == ss_id]
    if row.empty:
        raise ValueError(f"ss_id {ss_id} not found in {metadata_csv}")
    row = row.iloc[0]
    return {
        "ss_id": int(ss_id),
        "lat": float(row["latitude_rounded"]),
        "lon": float(row["longitude_rounded"]),
        "kwp": float(row["kWp"]),
        "tilt": float(row["tilt"]),                  # panel angle from flat, degrees
        "orientation": float(row["orientation"]),    # compass direction the panels face (180 = south)
    }


# ----------------------------------------------------------------------------------------------
# 2. PV readings: half-hourly Wh  ->  hourly kWh
# ----------------------------------------------------------------------------------------------
def read_pv(pv_dir: str | Path, ss_ids: list[int]) -> dict[int, pd.DataFrame]:
    """Raw readings of several sites, as {ss_id: table}. The folder holds one parquet file per
    month for all sites; the filter makes pandas read only the rows of the sites we asked for."""
    raw = pd.read_parquet(pv_dir, filters=[("ss_id", "in", [int(i) for i in ss_ids])])
    return {int(i): part for i, part in raw.groupby("ss_id")}


def _short_gaps(is_missing: np.ndarray, max_len: int) -> np.ndarray:
    """True for missing values that sit in a run of at most `max_len` and have data on both sides."""
    n = len(is_missing)
    result = np.zeros(n, dtype=bool)
    if n == 0 or max_len <= 0:
        return result
    padded = np.concatenate(([False], is_missing.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])  # positions where a run starts / ends
    for start, stop in zip(edges[0::2], edges[1::2]):  # each missing run is [start, stop)
        if (stop - start) <= max_len and start > 0 and stop < n:
            result[start:stop] = True
    return result


def clean_pv(raw: pd.DataFrame, lat: float, lon: float, max_gap_halfhours: int = 1) -> pd.Series:
    """Raw half-hourly Wh -> hourly kWh, indexed by the START of each hour (UTC).

    Assumption: a timestamp marks the END of its half-hour (the metadata supports this: days
    run from 00:30 to 00:00). Each stamp is moved to the start of its half-hour first.

    Missing values are handled like this:
      * night (sun below the horizon)            -> 0
      * a single missing daytime half-hour       -> linear interpolation
      * anything longer                          -> stays NaN (never silently invented)
    The timeline stays continuous (no days are dropped), so a row offset always equals a time
    offset and later steps can detect gaps."""
    # tidy the timestamps: UTC, no timezone, sorted
    times = pd.to_datetime(raw[TIME_COL])
    if getattr(times.dt, "tz", None) is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    s = pd.Series(raw[VALUE_COL].to_numpy(dtype="float64"), index=pd.DatetimeIndex(times))
    s = s.sort_index()

    # stamp = end of period -> move to the start of the period, then drop duplicate stamps
    s.index = s.index.round("30min") - pd.Timedelta(minutes=30)
    s = s[~s.index.duplicated(keep="first")]

    # a complete half-hourly timeline, whole days from first to last reading
    grid = pd.date_range(
        s.index.min().floor("D"),
        s.index.max().floor("D") + pd.Timedelta(days=1) - pd.Timedelta(minutes=30),
        freq="30min",
    )
    values = s.reindex(grid).to_numpy().copy()

    # night gaps -> 0
    night = solar_elevation(grid, lat, lon, offset_minutes=15) < 0
    values[np.isnan(values) & night] = 0.0

    # single daytime gaps -> interpolate
    interpolated = pd.Series(values, index=grid).interpolate(method="linear", limit_area="inside")
    fill = _short_gaps(np.isnan(values), max_gap_halfhours)
    values[fill] = interpolated.to_numpy()[fill]

    # hour = sum of its two half-hours (if one of them is missing, the hour is missing)
    hourly = pd.Series(values, index=grid).resample("1h", closed="left", label="left").sum(min_count=2)
    hourly = hourly.loc[hourly.first_valid_index():hourly.last_valid_index()] / 1000.0  # Wh -> kWh
    hourly.name = "generation_kwh"
    return hourly


# ----------------------------------------------------------------------------------------------
# 3. ERA5 weather
# ----------------------------------------------------------------------------------------------
def find_files(patterns: list[str]) -> list[Path]:
    """Expand patterns like 'data/raw/ERA5/*.csv' into a sorted list of files."""
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(p) for p in sorted(glob.glob(pattern)))
    return files


def _combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge frames with columns latitude, longitude, valid_time and some variables. This one rule
    covers both cases we have:
       * files with the SAME variables but different periods  -> stacked in time
       * files with DIFFERENT variables for the same period   -> placed side by side
    Rows with the same grid point and timestamp are merged, taking the first value in each column.
    Result: indexed by (latitude, longitude, valid_time), one row per grid point and hour."""
    both = pd.concat(frames)
    return both.groupby(["latitude", "longitude", "valid_time"]).first().sort_index()


def read_era5_csv(paths: list[Path]) -> pd.DataFrame:
    """ERA5 time-series CSVs (columns: latitude, longitude, valid_time, <variables>), ALL grid
    points. Read once; `weather_at` then picks the point nearest to each site."""
    if not paths:
        raise FileNotFoundError("No ERA5 CSV files found - check paths in config.yaml")
    return _combine([pd.read_csv(path, parse_dates=["valid_time"]) for path in paths])


def read_era5_grib(paths: list[Path], variables=("hcc", "mcc", "lcc")) -> pd.DataFrame:
    """Cloud-layer variables from GRIB files (needs xarray + cfgrib), ALL grid points, in the
    same shape as `read_era5_csv`."""
    import xarray as xr  # imported here so the CSV-only path works without it

    if not paths:
        raise FileNotFoundError("No GRIB files found - check paths in config.yaml")
    frames = []
    for path in paths:
        ds = xr.open_dataset(path, engine="cfgrib")
        try:
            for lat in np.atleast_1d(ds["latitude"].values):
                for lon in np.atleast_1d(ds["longitude"].values):
                    point = ds[list(variables)].sel(latitude=lat, longitude=lon)
                    frame = pd.DataFrame(
                        {v: np.asarray(point[v].values, dtype=float) for v in variables},
                        index=pd.DatetimeIndex(np.atleast_1d(point["time"].values), name="valid_time"),
                    )
                    frame["latitude"] = float(lat)
                    frame["longitude"] = float(((lon + 180) % 360) - 180)   # 0..360 -> -180..180
                    frames.append(frame.reset_index())
        finally:
            ds.close()
    return _combine(frames)


def weather_at(points: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """The hourly weather of the grid point nearest to (lat, lon), indexed by time."""
    grid = points.index.droplevel("valid_time").unique()
    nearest = min(grid, key=lambda p: (p[0] - lat) ** 2 + (p[1] - lon) ** 2)
    return points.loc[nearest].rename_axis("time")


# ----------------------------------------------------------------------------------------------
# 4. Line the weather up with the PV hours
# ----------------------------------------------------------------------------------------------
def align_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Express ERA5 for the hour [t, t+1h), the same hour a PV row describes.

    * accumulated variables (ssrd, fdir) are totals over the hour ENDING at their timestamp,
      so the total for [t, t+1h) is the value stamped t+1h; divided by 3600 s it becomes the
      average power in W/m2. Tiny negative values (numerical noise at night) are set to 0.
    * all other variables are snapshots, so we average the snapshots at t and t+1h."""
    w = weather.sort_index().astype(float)
    next_hour = w.copy()
    next_hour.index = next_hour.index - pd.Timedelta(hours=1)
    next_hour = next_hour.reindex(w.index)  # next_hour.loc[t] is the value stamped t+1h

    aligned = pd.DataFrame(index=w.index)
    for col in w.columns:
        if col in ACCUMULATED:
            aligned[col] = (next_hour[col] / 3600.0).clip(lower=0.0)
        else:
            aligned[col] = (w[col] + next_hour[col]) / 2.0
    return aligned


# ----------------------------------------------------------------------------------------------
# 5. Join PV and weather, add features
# ----------------------------------------------------------------------------------------------
def add_features(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """Calendar and sun-position columns. All of them can be computed for any future hour
    (no target information), so they are safe inputs for a forecast."""
    out = df.copy()
    # hour of day and day of year as circles: 23:00 and 00:00 are neighbours, not 23 apart
    hour = out.index.hour.to_numpy()
    doy = out.index.dayofyear.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    # sun height in the middle of each hour, and a day/night flag used only for the metrics
    out["solar_elevation"] = solar_elevation(out.index, lat, lon, offset_minutes=30)
    out["daylight"] = (out["solar_elevation"] > 0).astype("float32")
    return out


def assemble(pv_hourly: pd.Series, weather: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """One table, one row per hour, over the period where both PV and weather exist.
    Hours without PV data stay in the table as NaN, so gaps remain visible."""
    aligned = align_weather(weather)
    complete = aligned.dropna()  # hours where every weather variable is present
    start = max(pv_hourly.index.min(), complete.index.min())
    end = min(pv_hourly.index.max(), complete.index.max())
    timeline = pd.date_range(start, end, freq="1h")

    df = aligned.reindex(timeline)
    df.insert(0, "generation_kwh", pv_hourly.reindex(timeline))
    df["t2m_c"] = df.pop("t2m") - 273.15                 # kelvin -> Celsius
    df["ws10"] = np.hypot(df["u10"], df["v10"])           # wind speed from its two components
    df.index.name = "time"
    return add_features(df, lat, lon)


def add_site_columns(df: pd.DataFrame, site: dict) -> pd.DataFrame:
    """The target divided by the site's size, plus the site's fixed properties as input columns.

    yield_kwh_per_kwp : hourly output per installed kWp - THE TARGET for every model
    tilt              : panel angle (degrees)
    orient_sin/cos    : panel direction as a circle (so 359 and 1 degrees are neighbours)"""
    out = df.copy()
    out["yield_kwh_per_kwp"] = out["generation_kwh"] / site["kwp"]
    out["tilt"] = site["tilt"]
    out["orient_sin"] = np.sin(np.deg2rad(site["orientation"]))
    out["orient_cos"] = np.cos(np.deg2rad(site["orientation"]))
    return out


def complete_days(df: pd.DataFrame) -> pd.Series:
    """True for every calendar day that has a PV value for every daylight hour.
    (Night hours never count as missing.) Indexed by day."""
    ok = df["generation_kwh"].notna() | (df["daylight"] < 0.5)
    return ok.groupby(df.index.normalize()).all()


def month_coverage(df: pd.DataFrame, before: str | None = None) -> pd.Series:
    """Number of COMPLETE days per month of the year (index 1..12) for one site.
    `before` keeps only earlier rows, e.g. the training period. A site that has no complete
    day in some month cannot learn that season."""
    d = df if before is None else df.loc[: pd.Timestamp(before) - pd.Timedelta(hours=1)]
    full = complete_days(d)
    return full.groupby(full.index.month).sum().reindex(range(1, 13), fill_value=0)


# ----------------------------------------------------------------------------------------------
# Running the whole thing
# ----------------------------------------------------------------------------------------------
def build_all(cfg: dict, log=print) -> dict[int, tuple[pd.DataFrame, dict]]:
    """Build the table of every site in the site list: {ss_id: (table, site info)}."""
    paths = cfg["paths"]
    ids = [int(i) for i in pd.read_csv(cfg["sites_csv"])["ss_id"]]
    raw = read_pv(paths["pv_dir"], ids)
    csv_points = read_era5_csv(find_files(paths["era5_csv"]))        # weather is read only once
    grib_points = read_era5_grib(find_files(paths["era5_grib"])) if paths.get("era5_grib") else None

    result = {}
    for ss_id in ids:
        if ss_id not in raw:
            log(f"  site {ss_id}: no PV readings found, skipped")
            continue
        site = load_site_info(paths["metadata_csv"], ss_id)
        pv = clean_pv(raw[ss_id], site["lat"], site["lon"], cfg.get("max_gap_halfhours", 1))
        weather = weather_at(csv_points, site["lat"], site["lon"])
        if grib_points is not None:                                  # cloud layers: hours in both
            weather = weather.join(weather_at(grib_points, site["lat"], site["lon"]), how="inner")
        table = add_site_columns(assemble(pv, weather, site["lat"], site["lon"]), site)
        result[ss_id] = (table, site)
        log(f"  site {ss_id}: {len(table):,} hours, {table['generation_kwh'].notna().mean():.0%} with PV data")
    return result


def save_dataset(df: pd.DataFrame, site: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"site_{site['ss_id']}.parquet"
    df.to_parquet(path)
    path.with_suffix(".json").write_text(json.dumps(site, indent=2))
    return path


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    return pd.read_parquet(path), json.loads(path.with_suffix(".json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the clean hourly datasets for all sites.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    sites = build_all(cfg)
    rows = {}
    for ss_id, (df, site) in sites.items():
        save_dataset(df, site, cfg["paths"]["out_dir"])
        rows[ss_id] = month_coverage(df, before=cfg.get("test_start"))
    coverage = pd.DataFrame(rows).T.rename_axis("ss_id")
    coverage.columns = [f"m{m:02d}" for m in coverage.columns]
    coverage.to_csv(Path(cfg["paths"]["out_dir"]) / "month_coverage.csv")

    print(f"\nsaved {len(sites)} sites to {cfg['paths']['out_dir']}")
    weak = coverage[(coverage < cfg.get("min_days_per_month", 60)).any(axis=1)]
    print(f"complete days per calendar month before {cfg.get('test_start')}, weakest month per site:")
    print(coverage.min(axis=1).sort_values().head(10).to_string())
    print(f"\n{len(weak)} site(s) have a month with fewer than {cfg.get('min_days_per_month', 60)} complete days"
          + (f": {sorted(weak.index.tolist())}" if len(weak) else ""))


if __name__ == "__main__":
    main()
