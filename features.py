"""Shared cleaning and causal feature functions.

Used by analysis.ipynb, detect_anomalies.py and inference.py so training and
production apply exactly the same steps (see docs/design.md, Stage 4).

Rules:
    - Every step is causal: a value at time t only uses data up to t.
    - Learned parameters (temperature baseline) are fit on train only.
    - Labels are only used by the evaluation/EDA helpers at the bottom, never as features.
"""

import numpy as np
import pandas as pd

import config as cfg

# ============================================================
# Constants (fixed, not learned)
# ============================================================
HARDWARE_LIMITS = {          # physically possible ranges; outside = invalid reading
    cfg.TEMP: (-10.0, 50.0),
    cfg.POWER: (0.0, 2000.0),
    cfg.MOTION: (0, 1),
    cfg.DOOR: (0, 1),
}
FFILL_LIMIT_MINUTES = 2      # short gaps filled forward; longer gaps = sensor offline
WARMUP_MINUTES = 180         # history prepended from the previous split (longest lookback + margin)
TEMP_SMOOTH_MINUTES = 15     # rolling median before computing temperature rates

FRIDGE_IDLE_MAX_W = 50.0     # below: idle
FRIDGE_DEFROST_MIN_W = 200.0 # at/above: defrost; in between: running


# ============================================================
# Loading
# ============================================================
def load_split(split):
    """Raw readings of one split (motion/door as float so NaN is allowed)."""
    df = pd.read_csv(cfg.SPLIT_FILES[split], parse_dates=["timestamp"])
    return df.astype({s: float for s in cfg.SENSORS})


def load_labels():
    """Anomaly labels (evaluation / EDA only)."""
    return pd.read_csv(cfg.LABELS_FILE, parse_dates=["start", "end"])


def load_generation_log():
    """Benign events and data-quality issues (diagnostics only)."""
    return pd.read_csv(cfg.GENERATION_LOG_FILE, parse_dates=["start", "end"])


# ============================================================
# Cleaning
# ============================================================
def drop_duplicates_sorted(df):
    """Sort by time and keep the first row per timestamp."""
    df = df.sort_values("timestamp", kind="stable")
    return df.drop_duplicates("timestamp", keep="first")


def reindex_regular(df, start=None, end=None):
    """Regular 1-minute grid; missing rows (hub outage) become NaN rows.

    start/end: grid bounds [start, end). Default: first to last timestamp.
    """
    start = pd.Timestamp(start) if start is not None else df["timestamp"].min()
    end = pd.Timestamp(end) if end is not None else df["timestamp"].max() + pd.Timedelta(cfg.FREQ)
    grid = pd.date_range(start, end, freq=cfg.FREQ, inclusive="left", name="timestamp")
    return df.set_index("timestamp").reindex(grid)


def apply_hardware_limits(df):
    """Physically impossible values -> NaN. Outliers inside the limits are kept."""
    df = df.copy()
    for sensor, (lo, hi) in HARDWARE_LIMITS.items():
        df.loc[(df[sensor] < lo) | (df[sensor] > hi), sensor] = np.nan
    return df


def fill_short_gaps(df, limit=FFILL_LIMIT_MINUTES):
    """Forward-fill at most `limit` minutes (causal; no interpolation)."""
    df = df.copy()
    df[cfg.SENSORS] = df[cfg.SENSORS].ffill(limit=limit)
    return df


def add_offline_flags(df):
    """offline_<sensor> = True where the reading is still missing after filling."""
    df = df.copy()
    for sensor in cfg.SENSORS:
        df[f"offline_{sensor}"] = df[sensor].isna()
    return df


def clean(raw, start=None, end=None):
    """Full cleaning pipeline: dedupe -> regular grid -> limits -> short fill -> offline flags."""
    df = drop_duplicates_sorted(raw)
    df = reindex_regular(df, start, end)
    df = apply_hardware_limits(df)
    df = fill_short_gaps(df)
    return add_offline_flags(df)


def clean_split(split):
    """Load and clean one split on its configured date range."""
    start, end = cfg.SPLITS[split]
    return clean(load_split(split), start, end)


# ============================================================
# Warm-up from the previous split (past data only, as a live stream would have)
# ============================================================
def previous_split(split):
    """Name of the split just before this one (None for the first)."""
    order = list(cfg.SPLITS)
    i = order.index(split)
    return order[i - 1] if i > 0 else None


def clean_split_with_warmup(split, warmup=WARMUP_MINUTES):
    """Clean a split with the last `warmup` minutes of the previous split in front.

    The warm-up rows only give history to rolling/run-length features; they are
    dropped again in split_features. Nothing is fit on them.
    """
    start, end = cfg.SPLITS[split]
    prev = previous_split(split)
    if prev is None:
        return clean_split(split)
    warm_start = pd.Timestamp(start) - pd.Timedelta(minutes=warmup)
    prev_raw = load_split(prev)
    raw = pd.concat([prev_raw[prev_raw["timestamp"] >= warm_start], load_split(split)])
    return clean(raw, warm_start, end)


# ============================================================
# Data-quality report
# ============================================================
def count_runs(mask):
    """Number of consecutive True runs in a boolean series."""
    mask = mask.astype(bool)
    return int((mask & ~mask.shift(1, fill_value=False)).sum())


def data_quality_report(raw, cleaned):
    """One row of data-quality numbers for a split."""
    row = {
        "raw_rows": len(raw),
        "duplicate_rows": int(raw["timestamp"].duplicated().sum()),
        "missing_rows": len(cleaned) - raw["timestamp"].nunique(),
        "out_of_order": int((raw["timestamp"].diff() < pd.Timedelta(0)).sum()),
    }
    for sensor in cfg.SENSORS:
        offline = cleaned[f"offline_{sensor}"]
        row[f"{sensor}_missing_cells"] = int(raw[sensor].isna().sum())
        row[f"{sensor}_offline_min"] = int(offline.sum())
        row[f"{sensor}_offline_periods"] = count_runs(offline)
    return row


# ============================================================
# Time context
# ============================================================
def day_type(index):
    """tue_thu (house empty in office hours), mon_fri (WFH), weekend."""
    dow = index.day_of_week
    return np.select([dow.isin([1, 2, 3]), dow.isin([0, 4])], ["tue_thu", "mon_fri"], "weekend")


def baseline_group(day_types):
    """Two groups for the temperature baseline: tue_thu (eco mode) vs occupied (Mon/Fri + weekend)."""
    return np.where(np.asarray(day_types) == "tue_thu", "tue_thu", "occupied")


def add_time_context(df):
    """Hour of day, day type, and baseline group."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_type"] = day_type(df.index)
    df["baseline_group"] = baseline_group(df["day_type"])
    return df


# ============================================================
# Learned baseline (fit on train only)
# ============================================================
def fit_temp_baseline(train_clean):
    """Median temperature per (baseline_group, hour) on train. No labels used.

    Pooling Mon/Fri with weekends gives 8 train days per hour instead of 4, so a
    heating failure on one day contaminates at most a minority of each median.
    """
    ctx = add_time_context(train_clean)
    return ctx.groupby(["baseline_group", "hour"])[cfg.TEMP].median().rename("temp_baseline")


def lookup_temp_baseline(df, baseline):
    """Baseline value for every row (df must have baseline_group and hour)."""
    keys = pd.MultiIndex.from_arrays([df["baseline_group"], df["hour"]])
    return baseline.reindex(keys).to_numpy()


# ============================================================
# Per-sensor features (causal)
# ============================================================
def run_length(series):
    """Minutes the current value has persisted (NaN where the value is missing)."""
    block = (series != series.shift()).cumsum()
    length = series.groupby(block).cumcount() + 1
    return length.where(series.notna()).astype(float)


def temperature_features(df, baseline):
    """Deviation from the train baseline, residual from recent readings, and smoothed rate (degC/hour).

    - temp_dev: reading minus the hour/day baseline (context level).
    - temp_resid: reading minus the 15-min rolling median (isolates 1-3 min spikes in any context).
    - Rates compare 15-min rolling medians instead of single readings: cuts noise and ignores spikes.
    """
    temp = df[cfg.TEMP]
    smooth = temp.rolling(TEMP_SMOOTH_MINUTES, min_periods=10).median()
    baseline_values = lookup_temp_baseline(df, baseline)
    return pd.DataFrame({
        "temp_baseline": baseline_values,
        "temp_dev": temp - baseline_values,
        "temp_resid": temp - smooth,
        "temp_rate_15": (smooth - smooth.shift(15)) * (60 / 15),
        "temp_rate_60": (smooth - smooth.shift(60)),
    }, index=df.index)


def fridge_mode(power):
    """Fixed thresholds: idle / running / defrost (NaN when offline)."""
    mode = pd.Series(np.select([power < FRIDGE_IDLE_MAX_W, power < FRIDGE_DEFROST_MIN_W],
                               ["idle", "running"], "defrost"), index=power.index)
    return mode.where(power.notna())


def fridge_features(df):
    """Mode and minutes in the current mode."""
    mode = fridge_mode(df[cfg.POWER])
    return pd.DataFrame({"fridge_mode": mode, "fridge_mode_minutes": run_length(mode)}, index=df.index)


def motion_features(df):
    """Motion minutes in the last 15 and 60 minutes."""
    motion = df[cfg.MOTION]
    return pd.DataFrame({
        "motion_15": motion.rolling(15, min_periods=1).sum(),
        "motion_60": motion.rolling(60, min_periods=1).sum(),
    }, index=df.index)


def minutes_since(event, cap):
    """Minutes since `event` was last True (0 while True), capped at `cap`.

    No event within the history -> cap, so the value never depends on how much
    history happens to be loaded (keeps warm-up and continuous runs identical).
    """
    position = pd.Series(np.arange(len(event), dtype=float), index=event.index)
    last_seen = position.where(event.fillna(False).astype(bool)).ffill()
    return (position - last_seen).fillna(cap).clip(upper=cap)


def door_features(df):
    """Minutes the door has been open (0 when closed) and minutes since it was last open.

    minutes_since_door_open is NaN (unknown) if the door sensor was offline after its
    last opening: an entry during the outage would otherwise look like no entry.
    """
    door = df[cfg.DOOR]
    open_minutes = run_length(door).where(door == 1, 0.0)
    since_open = minutes_since(door == 1, cap=WARMUP_MINUTES)
    since_offline = minutes_since(door.isna(), cap=WARMUP_MINUTES)
    return pd.DataFrame({
        "door_open_minutes": open_minutes.where(door.notna()),
        "minutes_since_door_open": since_open.where(since_offline >= since_open),
    }, index=df.index)


def build_features(cleaned, baseline):
    """Cleaned readings + time context + all per-sensor features."""
    df = add_time_context(cleaned)
    return pd.concat([df, temperature_features(df, baseline), fridge_features(df),
                      motion_features(df), door_features(df)], axis=1)


def split_features(split, baseline):
    """Features for one split, computed with warm-up history and trimmed to the split."""
    start, end = cfg.SPLITS[split]
    feats = build_features(clean_split_with_warmup(split), baseline)
    return feats.loc[pd.Timestamp(start):]


# ============================================================
# Labels -> per-minute view (evaluation / EDA only, never a model input)
# ============================================================
def minute_labels(index, labels):
    """Per-minute anomaly_type and sensor_id (None = normal)."""
    out = pd.DataFrame({"anomaly_type": None, "anomaly_sensor": None}, index=index)
    for _, row in labels.iterrows():
        out.loc[row["start"]:row["end"], ["anomaly_type", "anomaly_sensor"]] = [row["anomaly_type"], row["sensor_id"]]
    return out
