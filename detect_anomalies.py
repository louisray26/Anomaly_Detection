"""Compare per-sensor detectors on val, select one, and save it.

Models (docs/design.md, Stage 5):
    rules       - physics / house-rule thresholds + user-configured away/night schedule
    iforest     - one Isolation Forest per sensor on causal features + generic time context
    iforest_v2  - fewer, sensor-specific inputs (learned activity rates instead of generic context)
    charts      - EWMA / CUSUM control charts on standardised signals (stats fit on train)

Pipeline:
    1. Features: train (fit) and val (tune/select); test is never loaded here
    2. Minute flags per sensor from each model
    3. Flags -> alerts (merge gap)
    4. Scoring against labels (hits, false alarms, cross-sensor, benign, delay)
    5. Tune per sensor on val, compare, select one model by total val errors
    6. Save selected model to artifacts/, alerts and metrics to outputs/

Run:  python detect_anomalies.py
"""

import itertools
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config as cfg
import features as F

SENSORS = cfg.SENSORS
BENIGN_KINDS = ["night_bathroom_trip", "weekend_guests"]


# ============================================================
# 1. Features
# ============================================================
RATE_COLS = {cfg.MOTION: "motion_usual", cfg.DOOR: "door_usual"}


def fit_activity_rates(train):
    """Share of active minutes per (day_type, hour) on train, Laplace-smoothed. No labels used."""
    rates = {}
    for sensor in RATE_COLS:
        g = train.groupby(["day_type", "hour"])[sensor]
        rates[sensor] = ((g.sum() + 0.5) / (g.count() + 1)).rename(RATE_COLS[sensor])
    return rates


def add_activity_rates(f, rates):
    """motion_usual / door_usual column for every row (learned context, not the schedule)."""
    f = f.copy()
    keys = pd.MultiIndex.from_arrays([f["day_type"], f["hour"]])
    for sensor, col in RATE_COLS.items():
        f[col] = rates[sensor].reindex(keys).to_numpy()
    return f


def load_features():
    """Train features (no warm-up available) and val features (warm-up from train)."""
    baseline = F.fit_temp_baseline(F.clean_split("train"))
    train = F.build_features(F.clean_split("train"), baseline)
    val = F.split_features("val", baseline)
    rates = fit_activity_rates(train)
    return baseline, rates, add_activity_rates(train, rates), add_activity_rates(val, rates)


# ============================================================
# 2a. Rule detector
# ============================================================
def in_window(index, window, days=None):
    """True where the minute of day falls in [start, end) (and weekday in days, if given).

    start > end means the window wraps midnight (e.g. 23:30-06:00).
    """
    minute = index.hour * 60 + index.minute
    if window[0] <= window[1]:
        inside = (minute >= window[0]) & (minute < window[1])
    else:
        inside = (minute >= window[0]) | (minute < window[1])
    if days is not None:
        inside &= index.day_of_week.isin(days)
    return pd.Series(inside, index=index)


def sustained(cond, minutes):
    """True once `cond` has held for `minutes` consecutive minutes (causal)."""
    return cond.astype(float).rolling(minutes, min_periods=minutes).sum() == minutes


def combine(*rules):
    """(flag, reason) per minute from several (condition, reason) pairs; first match wins."""
    index = rules[0][0].index
    flag = pd.Series(False, index=index)
    reason = pd.Series("", index=index, dtype=object)
    for cond, text in rules:
        cond = cond.fillna(False).astype(bool)
        new = cond & ~flag
        reason[new] = text
        flag |= cond
    return pd.DataFrame({"flag": flag, "reason": reason})


def rule_temperature(f, params, schedule):
    r = cfg.RULES
    spike = f["temp_resid"].abs() > r["spike_resid_degc"]
    falling = (f["temp_dev"] < -params["heating_dev_degc"]) & (f["temp_rate_60"] < -params["heating_rate_degc_h"])
    return combine(
        (spike, f"temp jumped > {r['spike_resid_degc']} degC from recent readings (spike)"),
        (sustained(falling, r["heating_sustain_minutes"]),
         f"temp {params['heating_dev_degc']} degC below usual and falling > {params['heating_rate_degc_h']} degC/h "
         f"for {r['heating_sustain_minutes']} min (heating failure)"),
    )


def rule_fridge(f, params, schedule):
    r = cfg.RULES
    no_power = sustained(f[cfg.POWER] < r["power_loss_watts"], r["power_loss_minutes"])
    stuck = (f["fridge_mode"] == "running") & (f["fridge_mode_minutes"] > r["stuck_on_minutes"])
    return combine(
        (no_power, f"fridge < {r['power_loss_watts']} W for {r['power_loss_minutes']} min (power loss)"),
        (stuck, f"compressor running > {r['stuck_on_minutes']} min (stuck on)"),
    )


def rule_motion(f, params, schedule):
    r = cfg.RULES
    away = in_window(f.index, schedule["away"]["window"], schedule["away"]["days"])
    intrusion = (away & (f[cfg.MOTION] == 1) & (f["motion_15"] >= r["intrusion_min_motion_15"])
                 & (f["minutes_since_door_open"] > params["intrusion_no_door_minutes"]))
    return combine((intrusion, f"motion in away hours, no door opening for "
                               f"{params['intrusion_no_door_minutes']} min (intrusion)"))


def rule_door(f, params, schedule):
    r = cfg.RULES
    night = in_window(f.index, schedule["night"]["window"])
    return combine(
        (night & (f[cfg.DOOR] == 1), "door open in night hours (night opening)"),
        (f["door_open_minutes"] > r["door_left_open_minutes"], f"door open > {r['door_left_open_minutes']} min (left open)"),
    )


RULE_FUNCS = {cfg.TEMP: rule_temperature, cfg.POWER: rule_fridge, cfg.MOTION: rule_motion, cfg.DOOR: rule_door}

# Tuned parameters per sensor (sensors not listed have none)
RULE_TUNED = {
    cfg.TEMP: ["heating_dev_degc", "heating_rate_degc_h"],
    cfg.MOTION: ["intrusion_no_door_minutes"],
}


def rule_flags(f, sensor, params, schedule=cfg.SCHEDULE):
    """Minute flags + reasons for one sensor."""
    return RULE_FUNCS[sensor](f, params, schedule)


# ============================================================
# 2b. Isolation Forest detector
# ============================================================
CONTEXT = ["hour_sin", "hour_cos", "dt_tue_thu", "dt_mon_fri", "dt_weekend"]
IF_INPUTS = {   # numeric inputs (used for the explanation) + one-hot / cyclic context
    cfg.TEMP: (["temp_dev", "temp_resid", "temp_rate_15", "temp_rate_60"], CONTEXT),
    cfg.POWER: ([cfg.POWER, "fridge_mode_minutes"], ["mode_idle", "mode_running", "mode_defrost"]),
    cfg.MOTION: (["motion_15", "motion_60", "minutes_since_door_open"], CONTEXT),
    cfg.DOOR: (["door_open_minutes"], CONTEXT),
}
# v2: 2-3 inputs per sensor; context only as a learned rate where the sensor needs it
IF_V2_INPUTS = {
    cfg.TEMP: (["temp_dev", "temp_resid", "temp_rate_60"], []),
    cfg.POWER: ([cfg.POWER, "fridge_mode_minutes"], []),
    cfg.MOTION: (["motion_15", "minutes_since_door_open", "motion_usual"], []),
    cfg.DOOR: (["door_open_minutes", "door_usual"], []),
}


def encode(f, inputs):
    """Model matrix for one sensor; rows with any missing input are dropped (detector skips them)."""
    enc = pd.DataFrame(index=f.index)
    angle = 2 * np.pi * f["hour"] / 24
    enc["hour_sin"], enc["hour_cos"] = np.sin(angle), np.cos(angle)
    for dt in ["tue_thu", "mon_fri", "weekend"]:
        enc[f"dt_{dt}"] = (f["day_type"] == dt).astype(float)
    for mode in ["idle", "running", "defrost"]:
        enc[f"mode_{mode}"] = (f["fridge_mode"] == mode).astype(float).where(f["fridge_mode"].notna())
    numeric, context = inputs
    X = pd.concat([f[numeric], enc[context]], axis=1)
    return X.dropna()


def fit_iforest(train, inputs_map, params):
    """One forest per sensor + train score distribution + train quantiles for explanations."""
    models = {}
    for sensor in SENSORS:
        X = encode(train, inputs_map[sensor])
        forest = IsolationForest(**params, random_state=cfg.SEED).fit(X)
        scores = -forest.score_samples(X)                       # higher = more anomalous
        numeric = inputs_map[sensor][0]
        models[sensor] = {
            "forest": forest,
            "inputs": inputs_map[sensor],
            "columns": list(X.columns),
            "score_percentiles": {p: float(np.percentile(scores, p)) for p in cfg.IF_PERCENTILE_GRID},
            "explain_quantiles": {c: np.quantile(X[c], np.linspace(0, 1, 1001)) for c in numeric},
        }
    return models


def iforest_scores(f, model, sensor):
    """Anomaly score per minute (NaN where the detector skips)."""
    X = encode(f, model["inputs"])[model["columns"]]
    scores = pd.Series(np.nan, index=f.index)
    scores[X.index] = -model["forest"].score_samples(X)
    return scores


def explain(f, model, sensor, minute):
    """Most extreme numeric input at this minute, as a percentile of train."""
    best, best_text = -1.0, ""
    for col, q in model["explain_quantiles"].items():
        value = f.at[minute, col]
        rank = np.searchsorted(q, value, side="right") / len(q)
        extremeness = abs(rank - 0.5)
        if extremeness > best:
            side = f"above {rank:.1%}" if rank >= 0.5 else f"below {1 - rank:.1%}"
            best, best_text = extremeness, f"{col} = {value:.2f}, {side} of train"
    return best_text


def iforest_flags(f, model, sensor, threshold):
    """Minute flags; reasons are filled per alert (first minute) to keep this fast."""
    scores = iforest_scores(f, model, sensor)
    return pd.DataFrame({"flag": (scores > threshold).fillna(False), "reason": ""}, index=f.index)


# ============================================================
# 2c. Control charts (EWMA / CUSUM)
# ============================================================
# (column, side, standardisation): None = one global median/MAD, list = median/MAD per group,
# "rate" = binary signal standardised by its learned per-(day_type, hour) rate
CHART_STREAMS = {
    cfg.TEMP: [("temp_dev", "lower", ["baseline_group", "hour"])],   # shifts only; spikes are a rule's job
    cfg.POWER: [(cfg.POWER, "both", ["fridge_mode"]), ("fridge_mode_minutes", "upper", ["fridge_mode"])],
    cfg.MOTION: [(cfg.MOTION, "upper", "rate")],
    cfg.DOOR: [(cfg.DOOR, "upper", "rate")],
}
SCALE_FLOOR = {"temp_dev": 0.1, cfg.POWER: 0.5, "fridge_mode_minutes": 1.0}


def robust_center_scale(x):
    center = x.median()
    return center, 1.4826 * (x - center).abs().median()


def fit_chart_stats(train):
    """Median / MAD per stream (and group) on unlabelled train; scale floored to avoid /0."""
    stats = {}
    for streams in CHART_STREAMS.values():
        for col, _, groups in streams:
            if groups == "rate":
                continue
            if groups is None:
                center, scale = robust_center_scale(train[col])
            else:
                g = train.groupby(groups)[col]
                center, scale = g.median(), g.apply(lambda x: robust_center_scale(x)[1])
            stats[col] = {"center": center, "scale": np.maximum(scale, SCALE_FLOOR[col])}
    return stats


def group_lookup(table, f, groups):
    keys = pd.Index(f[groups[0]]) if len(groups) == 1 else pd.MultiIndex.from_frame(f[groups])
    return table.reindex(keys).to_numpy(dtype=float)


def standardize(f, col, groups, stats):
    """z-score per minute (NaN where the reading or its group is missing)."""
    x = f[col].to_numpy(dtype=float)
    if groups == "rate":
        p = f[RATE_COLS[col]].to_numpy(dtype=float)
        return (x - p) / np.sqrt(p * (1 - p))
    s = stats[col]
    if groups is None:
        return (x - s["center"]) / s["scale"]
    return (x - group_lookup(s["center"], f, groups)) / group_lookup(s["scale"], f, groups)


def run_chart(z, chart, limit):
    """Upper and lower chart statistic per minute, in limit units. Missing minutes hold the state.

    CUSUM restarts at 0 after a signal (standard practice); otherwise one long anomaly
    builds a sum that stays above the limit for days.
    """
    up, lo = np.zeros(len(z)), np.zeros(len(z))
    lam, k = cfg.CHARTS["ewma_lambda"], cfg.CHARTS["cusum_k"]
    sigma = np.sqrt(lam / (2 - lam))                  # asymptotic EWMA sd for unit-variance z
    e = s_hi = s_lo = 0.0
    for i, v in enumerate(z):
        if not np.isnan(v):
            if chart == "ewma":
                e = lam * v + (1 - lam) * e
            else:
                s_hi, s_lo = max(0.0, s_hi + v - k), max(0.0, s_lo - v - k)
        up[i], lo[i] = (e / sigma, -e / sigma) if chart == "ewma" else (s_hi, s_lo)
        if chart == "cusum":
            s_hi, s_lo = (0.0 if s_hi > limit else s_hi), (0.0 if s_lo > limit else s_lo)
    return up, lo


def chart_statistics(f, sensor, stats, chart, limit):
    """Chart statistics for every stream of one sensor."""
    out = []
    for col, side, groups in CHART_STREAMS[sensor]:
        up, lo = run_chart(standardize(f, col, groups, stats), chart, limit)
        out.append((col, side, pd.Series(up, index=f.index), pd.Series(lo, index=f.index)))
    return out


def chart_flags_from_stats(chart_stats, chart, limit):
    """Flag + reason: first stream whose chart crosses the limit."""
    unit = "sigma" if chart == "ewma" else "(CUSUM h)"
    rules = []
    for col, side, up, lo in chart_stats:
        if side in ("upper", "both"):
            rules.append((up > limit, f"{chart.upper()} of {col} above +{limit} {unit}"))
        if side in ("lower", "both"):
            rules.append((lo > limit, f"{chart.upper()} of {col} below -{limit} {unit}"))
    return combine(*rules)


def chart_flags(f, sensor, stats, setting):
    chart, limit = setting
    return chart_flags_from_stats(chart_statistics(f, sensor, stats, chart, limit), chart, limit)


# ============================================================
# 3. Flags -> alerts
# ============================================================
def flags_to_alerts(flags, sensor, model_name, explain_fn=None):
    """Merge flagged minutes < gap apart into alerts; reason from the first minute."""
    flagged = flags.index[flags["flag"]]
    if len(flagged) == 0:
        return []
    gap = pd.Timedelta(minutes=cfg.ALERT_MERGE_GAP_MINUTES)
    group = (pd.Series(flagged).diff() >= gap).cumsum().to_numpy()
    alerts = []
    for g in np.unique(group):
        minutes = flagged[group == g]
        reason = explain_fn(minutes[0]) if explain_fn else flags.at[minutes[0], "reason"]
        alerts.append({"sensor": sensor, "model": model_name, "start": minutes[0], "end": minutes[-1], "reason": reason})
    return alerts


# ============================================================
# 4. Scoring
# ============================================================
def overlaps(a_start, a_end, b_start, b_end):
    return (a_start <= b_end) & (a_end >= b_start)


def score_sensor(alerts, labels, benign, sensor):
    """Hits, misses, false alarms (incl. cross-sensor and benign) and delays for one sensor."""
    grace = pd.Timedelta(minutes=cfg.GRACE_MINUTES)
    mine = [a for a in alerts if a["sensor"] == sensor]
    own = labels[labels["sensor_id"] == sensor]
    others = labels[labels["sensor_id"] != sensor]

    hits, delays, per_type = 0, [], {}
    for _, a in own.iterrows():
        matched = [al for al in mine if overlaps(al["start"], al["end"], a["start"], a["end"] + grace)]
        detected = bool(matched)
        hits += detected
        per_type.setdefault(a["anomaly_type"], [0, 0, []])      # hits, total, delays
        per_type[a["anomaly_type"]][0] += detected
        per_type[a["anomaly_type"]][1] += 1
        if detected:
            first = min(al["start"] for al in matched)
            delays.append(max(0.0, (first - a["start"]).total_seconds() / 60))
            per_type[a["anomaly_type"]][2].append(delays[-1])

    false_alarms, cross, on_benign = 0, 0, 0
    for al in mine:
        if any(overlaps(al["start"], al["end"], a["start"], a["end"] + grace) for _, a in own.iterrows()):
            continue
        false_alarms += 1
        cross += any(overlaps(al["start"], al["end"], a["start"], a["end"] + grace) for _, a in others.iterrows())
        on_benign += any(overlaps(al["start"], al["end"], b["start"], b["end"]) for _, b in benign.iterrows())

    return {
        "sensor": sensor, "anomalies": len(own), "hits": hits, "misses": len(own) - hits,
        "false_alarms": false_alarms, "cross_sensor_fa": cross, "benign_fa": on_benign,
        "errors": (len(own) - hits) + false_alarms,
        "median_delay_min": float(np.median(delays)) if delays else np.nan,
        "per_type": per_type,
    }


# ============================================================
# 5. Tuning on val (per sensor)
# ============================================================
def selection_key(result, conservativeness):
    """Fewest errors -> lowest median delay -> most conservative."""
    delay = result["median_delay_min"]
    return (result["errors"], np.inf if np.isnan(delay) else delay, -conservativeness)


def tune_rules(val, labels, benign, schedule=cfg.SCHEDULE):
    """Grid search per sensor; returns chosen params, per-sensor results, and the full grid log."""
    chosen, results, log = {}, {}, []
    for sensor in SENSORS:
        names = RULE_TUNED.get(sensor, [])
        grids = [cfg.RULE_GRID[n] for n in names]
        best = None
        for combo in itertools.product(*grids) if names else [()]:
            params = dict(zip(names, combo))
            # conservativeness: sum of grid positions (later = stricter)
            cons = sum(cfg.RULE_GRID[n].index(v) for n, v in params.items())
            alerts = flags_to_alerts(rule_flags(val, sensor, params, schedule), sensor, "rules")
            res = score_sensor(alerts, labels, benign, sensor)
            log.append({"model": "rules", "sensor": sensor, "setting": json.dumps(params), **{k: res[k] for k in
                        ["hits", "misses", "false_alarms", "errors", "median_delay_min"]}})
            if best is None or selection_key(res, cons) < selection_key(best[1], best[2]):
                best = (params, res, cons, alerts)
        chosen[sensor], results[sensor] = best[0], (best[1], best[3])
    return chosen, results, log


def tune_iforest(val, models, labels, benign, name="iforest"):
    """Choose a train-score percentile per sensor on val."""
    chosen, results, log = {}, {}, []
    for sensor in SENSORS:
        model = models[sensor]
        scores = iforest_scores(val, model, sensor)
        best = None
        for i, p in enumerate(cfg.IF_PERCENTILE_GRID):
            threshold = model["score_percentiles"][p]
            flags = pd.DataFrame({"flag": (scores > threshold).fillna(False), "reason": ""})
            alerts = flags_to_alerts(flags, sensor, name, lambda m: explain(val, model, sensor, m))
            res = score_sensor(alerts, labels, benign, sensor)
            log.append({"model": name, "sensor": sensor, "setting": f"p{p}", **{k: res[k] for k in
                        ["hits", "misses", "false_alarms", "errors", "median_delay_min"]}})
            if best is None or selection_key(res, i) < selection_key(best[1], best[2]):
                best = (p, res, i, alerts)
        chosen[sensor], results[sensor] = best[0], (best[1], best[3])
    return chosen, results, log


def tune_charts(val, stats, labels, benign):
    """Choose chart type (EWMA / CUSUM) and limit per sensor on val."""
    chosen, results, log = {}, {}, []
    for sensor in SENSORS:
        best = None
        for chart, limits in cfg.CHART_GRID.items():
            for i, limit in enumerate(limits):
                chart_stats = chart_statistics(val, sensor, stats, chart, limit)
                alerts = flags_to_alerts(chart_flags_from_stats(chart_stats, chart, limit), sensor, "charts")
                res = score_sensor(alerts, labels, benign, sensor)
                log.append({"model": "charts", "sensor": sensor, "setting": f"{chart} {limit}", **{k: res[k] for k in
                            ["hits", "misses", "false_alarms", "errors", "median_delay_min"]}})
                if best is None or selection_key(res, i) < selection_key(best[1], best[2]):
                    best = ((chart, limit), res, i, alerts)
        chosen[sensor], results[sensor] = best[0], (best[1], best[3])
    return chosen, results, log


def chart_reaction_table(log):
    """Best setting per sensor for each chart type (how each sensor reacts to EWMA vs CUSUM)."""
    df = pd.DataFrame([r for r in log if r["model"] == "charts"])
    df["chart"] = df["setting"].str.split().str[0]
    df["delay"] = df["median_delay_min"].fillna(np.inf)
    df["limit"] = df["setting"].str.split().str[1].astype(float)
    best = df.sort_values(["errors", "delay", "limit"], ascending=[True, True, False]).groupby(["sensor", "chart"], sort=False).head(1)
    return best.drop(columns=["model", "chart", "delay", "limit"]).sort_values(["sensor", "setting"])


# ============================================================
# 6. Comparison, selection and saving
# ============================================================
def time_per_minute(fn, n_minutes, repeats=3):
    """Mean detector time per input minute (ms), features already built."""
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return 1000 * (time.perf_counter() - start) / repeats / n_minutes


def summarise(name, results, n_days, size_kb, ms_per_minute, n_tuned, explainability):
    """One overall row for the comparison table."""
    rows = [r for r, _ in results.values()]
    delays = [r["median_delay_min"] for r in rows if not np.isnan(r["median_delay_min"])]
    fa = sum(r["false_alarms"] for r in rows)
    return {
        "model": name,
        "detected": f"{sum(r['hits'] for r in rows)}/{sum(r['anomalies'] for r in rows)}",
        "misses": sum(r["misses"] for r in rows), "false_alarms": fa,
        "fa_per_day": round(fa / n_days, 2),
        "cross_sensor_fa": sum(r["cross_sensor_fa"] for r in rows),
        "benign_fa": sum(r["benign_fa"] for r in rows),
        "total_errors": sum(r["errors"] for r in rows),
        "median_delay_min (median over sensors)": float(np.median(delays)) if delays else np.nan,
        "model_size_kb": round(size_kb, 1), "ms_per_minute": round(ms_per_minute, 4),
        "tuned_params": n_tuned, "explainability": explainability,
    }


def per_sensor_table(name, results):
    rows = []
    for sensor, (r, _) in results.items():
        rows.append({"model": name, **{k: v for k, v in r.items() if k != "per_type"}})
    return pd.DataFrame(rows)


def per_type_table(name, results):
    rows = []
    for sensor, (r, _) in results.items():
        for atype, (hit, total, delays) in r["per_type"].items():
            rows.append({"model": name, "sensor": sensor, "anomaly_type": atype, "detected": f"{hit}/{total}",
                         "median_delay_min": float(np.median(delays)) if delays else np.nan})
    return pd.DataFrame(rows)


def artifact_size_kb(obj, path):
    joblib.dump(obj, path)
    return path.stat().st_size / 1024


# ============================================================
# Main
# ============================================================
def main():
    cfg.ARTIFACTS_DIR.mkdir(exist_ok=True)
    cfg.OUTPUTS_DIR.mkdir(exist_ok=True)

    # Data (test is never loaded here)
    baseline, rates, train, val = load_features()
    labels = F.load_labels()
    labels = labels[labels["split"] == "val"]
    log = F.load_generation_log()
    benign = log[(log["split"] == "val") & log["kind"].isin(BENIGN_KINDS)]
    n_days = len(val) / (24 * 60)
    candidates, results, tuning_log = {}, {}, []

    # Rules: tune on val
    rule_params, results["rules"], lg = tune_rules(val, labels, benign)
    tuning_log += lg
    candidates["rules"] = {"model": "rules", "params": rule_params, "fixed": cfg.RULES, "schedule": cfg.SCHEDULE,
                           "baseline": baseline}
    timing = {"rules": lambda: [rule_flags(val, s, rule_params[s]) for s in SENSORS]}

    # Sensitivity: rules with a wider schedule only (not used for selection)
    wide_params, wide_results, _ = tune_rules(val, labels, benign, cfg.SCHEDULE_SENSITIVITY)
    sensitivity = pd.concat([per_sensor_table("rules (main schedule)", results["rules"]),
                             per_sensor_table("rules (wide schedule)", wide_results)])

    # Isolation Forests (v1, v2): fit on unlabelled train, tune threshold on val
    for name, inputs, params in [("iforest", IF_INPUTS, cfg.IFOREST), ("iforest_v2", IF_V2_INPUTS, cfg.IFOREST_V2)]:
        models = fit_iforest(train, inputs, params)
        pcts, results[name], lg = tune_iforest(val, models, labels, benign, name)
        tuning_log += lg
        thresholds = {s: models[s]["score_percentiles"][pcts[s]] for s in SENSORS}
        candidates[name] = {"model": name, "models": models, "percentiles": pcts, "thresholds": thresholds,
                            "baseline": baseline, "activity_rates": rates}
        timing[name] = lambda m=models, t=thresholds: [iforest_flags(val, m[s], s, t[s]) for s in SENSORS]

    # Control charts: centre/scale fit on unlabelled train, chart type + limit tuned on val
    chart_stats = fit_chart_stats(train)
    chart_settings, results["charts"], lg = tune_charts(val, chart_stats, labels, benign)
    tuning_log += lg
    candidates["charts"] = {"model": "charts", "settings": chart_settings, "stats": chart_stats,
                            "constants": cfg.CHARTS, "baseline": baseline, "activity_rates": rates}
    timing["charts"] = lambda: [chart_flags(val, s, chart_stats, chart_settings[s]) for s in SENSORS]

    # Comparison
    n_tuned = {"rules": 3, "iforest": 4, "iforest_v2": 4, "charts": 8}
    explainability = {
        "rules": "rule name + threshold (e.g. 'compressor running > 30 min')",
        "iforest": "most extreme feature vs train (e.g. 'fridge_mode_minutes = 85, above 99.9%')",
        "iforest_v2": "most extreme feature vs train",
        "charts": "chart + signal + side (e.g. 'CUSUM of temp_dev below -20')",
    }
    rows = []
    for name in candidates:
        size = artifact_size_kb(candidates[name], cfg.ARTIFACTS_DIR / f"candidate_{name}.joblib")
        ms = time_per_minute(timing[name], len(val))
        rows.append(summarise(name, results[name], n_days, size, ms, n_tuned[name], explainability[name]))
    comparison = pd.DataFrame(rows)
    per_sensor = pd.concat([per_sensor_table(n, r) for n, r in results.items()])
    per_type = pd.concat([per_type_table(n, r) for n, r in results.items()])
    reactions = chart_reaction_table(tuning_log)

    # Selection: fewest total errors; tie -> simpler model (order below)
    simplicity = {"rules": 0, "charts": 1, "iforest_v2": 2, "iforest": 3}
    selected = min(rows, key=lambda r: (r["total_errors"], simplicity[r["model"]]))["model"]
    joblib.dump(candidates[selected], cfg.ARTIFACTS_DIR / "selected_model.joblib")
    settings = {"rules": rule_params, "charts": {s: f"{c} {l}" for s, (c, l) in chart_settings.items()}}
    settings = settings.get(selected) or {s: f"p{p}" for s, p in candidates[selected]["percentiles"].items()}
    with open(cfg.ARTIFACTS_DIR / "selected_model.json", "w") as fh:
        json.dump({"model": selected, "tuned_settings": settings}, fh, indent=2)

    # Outputs: alerts (the task's "detected anomalies") and metrics
    for name, res in results.items():
        alerts = [a for _, (_, al) in res.items() for a in al]
        pd.DataFrame(alerts).sort_values("start").to_csv(cfg.OUTPUTS_DIR / f"alerts_{name}_val.csv", index=False)
    comparison.to_csv(cfg.OUTPUTS_DIR / "val_comparison.csv", index=False)
    per_sensor.to_csv(cfg.OUTPUTS_DIR / "val_per_sensor.csv", index=False)
    per_type.to_csv(cfg.OUTPUTS_DIR / "val_per_type.csv", index=False)
    reactions.to_csv(cfg.OUTPUTS_DIR / "val_chart_reactions.csv", index=False)
    sensitivity.to_csv(cfg.OUTPUTS_DIR / "val_sensitivity.csv", index=False)
    pd.DataFrame(tuning_log).to_csv(cfg.OUTPUTS_DIR / "val_tuning_log.csv", index=False)

    # Report
    pd.set_option("display.width", 220)
    print("Tuned rules:", rule_params)
    for name in ["iforest", "iforest_v2"]:
        print(f"Tuned {name} percentiles:", candidates[name]["percentiles"])
    print("Tuned charts:", chart_settings)
    print("\nVal comparison:\n", comparison.drop(columns="explainability").to_string(index=False))
    print("\nVal per sensor:\n", per_sensor.drop(columns=["anomalies", "per_type"], errors="ignore").to_string(index=False))
    print("\nVal per anomaly type (detected | median delay min):\n",
          per_type.pivot(index=["sensor", "anomaly_type"], columns="model",
                         values=["detected", "median_delay_min"]).to_string())
    print("\nControl charts, best setting per sensor and chart:\n", reactions.to_string(index=False))
    print("\nSensitivity, rules with wider schedule (tuned:", wide_params, "):\n", sensitivity.to_string(index=False))
    print(f"\nSelected model: {selected}  -> artifacts/selected_model.joblib")


if __name__ == "__main__":
    main()
