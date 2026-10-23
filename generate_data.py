"""Generate a synthetic 28-day smart-home sensor dataset with injected anomalies.

Pipeline:
    1. Simulate household occupancy (who is home / asleep, routine door use)
    2. Plan all injections up front: outages, benign events, anomalies
    3. Simulate normal sensor behaviour (heating failure is part of the physics)
    4. Overlay benign events and anomalies
    5. Apply data-quality issues (outages, dropped readings, duplicates)
    6. Split chronologically and save readings, labels and generation log

Outputs (see config.py for paths):
    data/train.csv, data/val.csv, data/test.csv   - sensor readings only
    data/anomaly_labels.csv                        - ground truth, never a model input
    data/generation_log.csv                        - benign events + data-quality issues

Run:  python generate_data.py
"""

import numpy as np
import pandas as pd

import config as cfg

MIN_PER_DAY = 1440


# ============================================================
# 1. Timeline helpers
# ============================================================
def build_index():
    """Regular 1-minute timestamps for the whole period."""
    return pd.date_range(cfg.START, periods=cfg.N_DAYS * MIN_PER_DAY, freq=cfg.FREQ)


def split_bounds(index):
    """Minute positions [start, end) of each split."""
    return {name: (index.searchsorted(pd.Timestamp(s)), index.searchsorted(pd.Timestamp(e)))
            for name, (s, e) in cfg.SPLITS.items()}


def time_jitter(rng):
    """Daily schedule jitter in minutes (clipped normal)."""
    occ = cfg.OCCUPANCY
    return int(np.clip(rng.normal(0, occ["jitter_sd"]), -occ["jitter_max"], occ["jitter_max"]))


def randint(rng, low_high):
    """Random integer in an inclusive (low, high) range."""
    return int(rng.integers(low_high[0], low_high[1], endpoint=True))


def dilate(mask, margin):
    """True wherever a True value lies within +/- margin minutes."""
    return pd.Series(mask).rolling(2 * margin + 1, center=True, min_periods=1).max().to_numpy() > 0


def erode(mask, margin):
    """True only where the whole +/- margin neighbourhood is True."""
    return ~dilate(~mask, margin)


# ============================================================
# 2. Occupancy schedule
# ============================================================
def simulate_occupancy(index, rng):
    """Per-minute occupancy state plus the minutes when someone leaves/arrives.

    Adult A: office Mon-Fri, one weekend outing per day.
    Adult B: office Tue-Thu (hybrid), always home at weekends.
    """
    occ = cfg.OCCUPANCY
    n = len(index)
    home_a = np.ones(n, dtype=bool)
    home_b = np.ones(n, dtype=bool)
    asleep = np.zeros(n, dtype=bool)
    trips = []                  # minute positions of leave / arrive door openings
    sleep_start = 0             # the period starts at midnight, asleep

    for day in range(cfg.N_DAYS):
        day0 = day * MIN_PER_DAY
        weekday = index[day0].day_of_week

        # Sleep / wake
        wake = day0 + occ["wake_mean"] + time_jitter(rng)
        asleep[sleep_start:wake] = True
        sleep_start = day0 + occ["sleep_mean"] + time_jitter(rng)

        # Office days
        for home, office_days in ((home_a, occ["office_days_a"]), (home_b, occ["office_days_b"])):
            if weekday in office_days:
                leave = day0 + randint(rng, occ["leave_window"])
                back = day0 + randint(rng, occ["return_window"])
                home[leave:back] = False
                trips += [leave, back]

        # Weekend outing (adult A only, adult B stays home)
        if weekday >= 5:
            hours = rng.uniform(*occ["weekend_outing_hours"])
            dur = int(hours * 60)
            w0, w1 = occ["weekend_outing_window"]
            start = day0 + int(rng.integers(w0, w1 - dur))
            home_a[start:start + dur] = False
            trips += [start, start + dur]

    asleep[sleep_start:] = True

    occupancy = pd.DataFrame({
        "n_home": home_a.astype(int) + home_b.astype(int),
        "asleep": asleep,
    })
    occupancy["awake_home"] = (occupancy["n_home"] > 0) & ~occupancy["asleep"]
    return occupancy, sorted(trips)


def add_extra_door_trips(occupancy, trips, rng):
    """0-2 extra door openings per day while someone is awake and home."""
    awake_home = occupancy["awake_home"].to_numpy()
    near_trip = np.zeros(len(occupancy), dtype=bool)
    near_trip[trips] = True
    allowed = awake_home & ~dilate(near_trip, 10)

    extra = []
    for day in range(cfg.N_DAYS):
        day0 = day * MIN_PER_DAY
        candidates = np.flatnonzero(allowed[day0:day0 + MIN_PER_DAY]) + day0
        k = randint(rng, cfg.OCCUPANCY["extra_door_openings"])
        if k and len(candidates):
            extra += list(rng.choice(candidates, size=k, replace=False))
    return sorted(trips + [int(t) for t in extra])


# ============================================================
# 3. Injection planning (spans are [start, end) minute positions)
# ============================================================
class SpanPlanner:
    """Places non-overlapping spans: anomalies keep a larger gap than other spans."""

    def __init__(self, rng):
        self.rng = rng
        self.anomaly_spans = []
        self.other_spans = []

    def _is_free(self, start, end, is_anomaly):
        def clear(spans, gap):
            return all(end + gap <= s or start >= e + gap for s, e in spans)
        if is_anomaly:
            return (clear(self.anomaly_spans, cfg.ANOMALY_GAP_MINUTES)
                    and clear(self.other_spans, cfg.RESERVED_BUFFER_MINUTES))
        return clear(self.anomaly_spans + self.other_spans, cfg.RESERVED_BUFFER_MINUTES)

    def place(self, allowed, lo, hi, dur, is_anomaly, reserve_after=0, tries=5000):
        """Random span of `dur` minutes inside [lo, hi) where `allowed` holds throughout.

        reserve_after: extra minutes kept free after the span (e.g. heating recovery).
        """
        candidates = np.flatnonzero(allowed[lo:hi - dur - reserve_after]) + lo
        for _ in range(tries):
            start = int(self.rng.choice(candidates))
            end = start + dur
            if allowed[start:end].all() and self._is_free(start, end + reserve_after, is_anomaly):
                (self.anomaly_spans if is_anomaly else self.other_spans).append((start, end + reserve_after))
                return start, end
        raise RuntimeError(f"Could not place a {dur}-minute span in [{lo}, {hi})")


def plan_outages(planner, bounds, n):
    """1 sensor outage per sensor and 1 hub outage per split."""
    dq = cfg.DATA_QUALITY
    anywhere = np.ones(n, dtype=bool)
    outages = []
    for split, (lo, hi) in bounds.items():
        for sensor in cfg.SENSORS:
            dur = randint(planner.rng, dq["sensor_outage_minutes"])
            s, e = planner.place(anywhere, lo, hi, dur, is_anomaly=False)
            outages.append({"kind": "sensor_outage", "sensor_id": sensor, "start": s, "end": e, "split": split})
        dur = randint(planner.rng, dq["hub_outage_minutes"])
        s, e = planner.place(anywhere, lo, hi, dur, is_anomaly=False)
        outages.append({"kind": "hub_outage", "sensor_id": "all", "start": s, "end": e, "split": split})
    return outages


def plan_benign_events(planner, index, bounds, occupancy):
    """Night bathroom trips (~3/week) and one weekend-guests evening per split."""
    rng = planner.rng
    tod = np.arange(len(index)) % MIN_PER_DAY
    asleep = occupancy["asleep"].to_numpy()
    events = []

    trip = cfg.BENIGN["night_bathroom_trip"]
    w0, w1 = trip["window"]
    night = asleep & (tod >= w0) & (tod < w1)
    for split, (lo, hi) in bounds.items():
        n_trips = round(trip["per_week"] * (hi - lo) / (7 * MIN_PER_DAY))
        for _ in range(n_trips):
            s, e = planner.place(night, lo, hi, randint(rng, trip["minutes"]), is_anomaly=False)
            events.append({"kind": "night_bathroom_trip", "sensor_id": cfg.MOTION, "start": s, "end": e, "split": split})

    guests = cfg.BENIGN["weekend_guests"]
    w0, w1 = guests["window"]
    saturday_evening = (index.day_of_week == 5) & (tod >= w0) & (tod < w1)
    for split, (lo, hi) in bounds.items():
        for _ in range(guests["per_split"]):
            dur = int(rng.uniform(*guests["hours"]) * 60)
            s, e = planner.place(saturday_evening, lo, hi, dur, is_anomaly=False)
            events.append({"kind": "weekend_guests", "sensor_id": f"{cfg.MOTION};{cfg.DOOR}",
                           "start": s, "end": e, "split": split})
    return events


def anomaly_allowed_masks(index, occupancy, target, door_trips):
    """Where each anomaly type may occur (the whole span must satisfy it)."""
    n = len(index)
    tod = np.arange(n) % MIN_PER_DAY
    anywhere = np.ones(n, dtype=bool)

    empty = occupancy["n_home"].to_numpy() == 0
    margin = cfg.ANOMALIES["intrusion_motion"]["empty_margin_minutes"]

    w0, w1 = cfg.ANOMALIES["door_open_night"]["window"]
    near_door_trip = np.zeros(n, dtype=bool)
    near_door_trip[door_trips] = True
    near_door_trip = dilate(near_door_trip, 15)

    # Heating failure: target must stay at 21 degC for the failure AND the recovery after it
    heating_on = pd.Series(target == cfg.TEMPERATURE["target_home"])
    reserve = cfg.ANOMALIES["heating_failure"]["recovery_reserve_minutes"]
    heating_on_ahead = heating_on[::-1].rolling(reserve + 1, min_periods=1).min()[::-1].to_numpy() > 0

    return {
        "temp_spike": anywhere,
        "heating_failure": heating_on_ahead,
        "fridge_power_loss": anywhere,
        "compressor_stuck_on": anywhere,
        "intrusion_motion": erode(empty, margin),
        "door_open_night": occupancy["asleep"].to_numpy() & (tod >= w0) & (tod < w1),
        "door_left_open": occupancy["awake_home"].to_numpy() & ~near_door_trip,
    }


def sample_anomaly_shape(atype, rng):
    """Duration (minutes) and magnitude for one anomaly."""
    spec = cfg.ANOMALIES[atype]
    if atype == "heating_failure":
        # Rate first, then a duration long enough for the minimum total drop
        rate = rng.uniform(*spec["magnitude"])
        min_hours = max(spec["hours"][0], spec["min_total_drop"] / rate)
        return int(np.ceil(rng.uniform(min_hours, spec["hours"][1]) * 60)), rate
    if "hours" in spec:
        dur = int(rng.uniform(*spec["hours"]) * 60)
    else:
        dur = randint(rng, spec["minutes"])
    if atype == "temp_spike":
        return dur, rng.uniform(*spec["magnitude"]) * rng.choice([-1, 1])
    if atype in ("door_open_night", "door_left_open"):
        return dur, dur
    return dur, rng.uniform(*spec["magnitude"])


def plan_anomalies(planner, bounds, masks):
    """Exact count per type per split; most constrained types are placed first."""
    order = ["heating_failure", "intrusion_motion", "door_open_night", "door_left_open",
             "compressor_stuck_on", "fridge_power_loss", "temp_spike"]
    anomalies = []
    for split, (lo, hi) in bounds.items():
        for atype in order:
            reserve = cfg.ANOMALIES[atype].get("recovery_reserve_minutes", 0)
            for _ in range(cfg.ANOMALY_COUNTS[split]):
                dur, magnitude = sample_anomaly_shape(atype, planner.rng)
                s, e = planner.place(masks[atype], lo, hi, dur, is_anomaly=True, reserve_after=reserve)
                anomalies.append({"anomaly_type": atype, "sensor_id": cfg.ANOMALIES[atype]["sensor"],
                                  "start": s, "end": e, "magnitude": magnitude,
                                  "magnitude_unit": cfg.ANOMALIES[atype]["unit"], "split": split})
    return anomalies


# ============================================================
# 4. Normal sensor simulation
# ============================================================
def thermostat_target(occupancy):
    """21 degC awake & home, 18 degC asleep, 18 degC eco mode when empty."""
    t = cfg.TEMPERATURE
    return np.where(occupancy["asleep"], t["target_night"],
                    np.where(occupancy["n_home"] == 0, t["target_eco"], t["target_home"]))


def solar_gain(tod):
    """Half-sine afternoon warming."""
    t = cfg.TEMPERATURE
    w0, w1 = t["solar_window"]
    phase = np.clip((tod - w0) / (w1 - w0), 0, 1)
    return t["solar_gain"] * np.sin(np.pi * phase)


def temperature_physics(target, solar, fall_rate):
    """Noise-free room temperature.

    First-order lag towards target + solar gain: fast when heating, slow when cooling.
    Heating failure overrides with a steady fall (fall_rate in degC/hour, NaN = heating works).
    """
    t = cfg.TEMPERATURE
    alpha_heat = 1.0 / t["lag_heat_minutes"]
    alpha_cool = 1.0 / t["lag_cool_minutes"]
    temp = np.empty(len(target))
    current = target[0]
    for i in range(len(target)):
        if np.isnan(fall_rate[i]):
            goal = target[i] + solar[i]
            current += (goal - current) * (alpha_heat if goal > current else alpha_cool)
        else:
            current -= fall_rate[i] / 60.0
        temp[i] = current
    return temp


def add_sensor_noise(values, sd, rng):
    """Gaussian measurement noise."""
    return values + rng.normal(0, sd, len(values))


def simulate_fridge(tod, n_home, rng):
    """Compressor on/off cycles, longer runs at mealtimes, periodic defrost."""
    f = cfg.FRIDGE
    n = len(tod)
    power = np.empty(n)
    running = np.zeros(n, dtype=bool)        # compressor or defrost heater on
    next_defrost = int(rng.uniform(*f["defrost_every_hours"]) * 60)
    i = 0
    while i < n:
        if i >= next_defrost:
            power[i:i + f["defrost_minutes"]] = rng.uniform(*f["defrost_power"])
            running[i:i + f["defrost_minutes"]] = True
            i += f["defrost_minutes"]
            next_defrost = i + int(rng.uniform(*f["defrost_every_hours"]) * 60)
            continue
        on = randint(rng, f["on_minutes"])
        is_meal = any(w0 <= tod[i] < w1 for w0, w1 in f["meal_windows"])
        if is_meal and n_home[i] > 0:
            on = round(on * f["meal_on_factor"])
        power[i:i + on] = rng.uniform(*f["on_power"])
        running[i:i + on] = True
        i += on
        off = randint(rng, f["off_minutes"])
        power[i:i + off] = rng.uniform(*f["off_power"])
        i += off

    # Noise depends on state: larger when running, small at idle
    noise_sd = np.where(running, f["noise_sd_on"], f["noise_sd_off"])
    return np.clip(power + rng.normal(0, 1, n) * noise_sd, 0, None)


def motion_probability(tod, occupancy):
    """Per-minute motion probability from occupancy and time of day."""
    m = cfg.MOTION_PROB
    morning = (tod >= m["morning_window"][0]) & (tod < m["morning_window"][1])
    evening = (tod >= m["evening_window"][0]) & (tod < m["evening_window"][1])
    home_p = np.where(morning, m["home_morning"], np.where(evening, m["home_evening"], m["home_base"]))
    return np.where(occupancy["asleep"], m["asleep"],
                    np.where(occupancy["n_home"] == 0, m["away"], home_p))


def open_door(door, motion, start, dur, rng, n_home):
    """Door open for `dur` minutes with hallway motion in its first and last minute (walk through, close).

    Motion before opening only if someone was already inside (not on arrival to an empty house);
    motion after closing only if someone is inside afterwards.
    """
    k_before = randint(rng, cfg.OCCUPANCY["motion_around_door"])
    k_after = randint(rng, cfg.OCCUPANCY["motion_around_door"])
    end = start + dur
    door[start:end] = 1
    motion[start] = 1
    motion[end - 1] = 1
    if start > 0 and n_home[start - 1] > 0:
        motion[max(start - k_before, 0):start] = 1
    if end < len(n_home) and n_home[end] > 0:
        motion[end:end + k_after] = 1


def simulate_motion_and_door(tod, occupancy, door_trips, rng):
    """Random activity motion plus routine door openings (leave, arrive, extras)."""
    n = len(tod)
    n_home = occupancy["n_home"].to_numpy()
    motion = (rng.random(n) < motion_probability(tod, occupancy)).astype(int)
    door = np.zeros(n, dtype=int)
    for t in door_trips:
        open_door(door, motion, t, randint(rng, cfg.OCCUPANCY["door_open_minutes"]), rng, n_home)
    return motion, door


# ============================================================
# 5. Overlay benign events and anomalies
# ============================================================
def apply_benign_events(readings, events, n_home, rng):
    """Bathroom trips add night motion; guests add door openings and busy motion."""
    for ev in events:
        s, e = ev["start"], ev["end"]
        if ev["kind"] == "night_bathroom_trip":
            readings[cfg.MOTION][s:e] = 1
        elif ev["kind"] == "weekend_guests":
            g = cfg.BENIGN["weekend_guests"]
            busy = rng.random(e - s) < rng.uniform(*g["motion_prob"])
            readings[cfg.MOTION][s:e] = np.maximum(readings[cfg.MOTION][s:e], busy)
            for t in rng.choice(np.arange(s, e - 5), size=randint(rng, g["door_openings"]), replace=False):
                open_door(readings[cfg.DOOR], readings[cfg.MOTION], int(t),
                          randint(rng, cfg.OCCUPANCY["door_open_minutes"]), rng, n_home)


def apply_anomalies(readings, anomalies, n_home, rng):
    """Overlay every anomaly except heating failure (handled in the temperature physics)."""
    for a in anomalies:
        s, e, mag = a["start"], a["end"], a["magnitude"]
        atype = a["anomaly_type"]
        if atype == "temp_spike":
            readings[cfg.TEMP][s:e] += mag
        elif atype == "fridge_power_loss":
            readings[cfg.POWER][s:e] = np.clip(mag + rng.normal(0, 0.05, e - s), 0, None)
        elif atype == "compressor_stuck_on":
            readings[cfg.POWER][s:e] = mag + rng.normal(0, cfg.FRIDGE["noise_sd_on"], e - s)
        elif atype == "intrusion_motion":
            moves = (rng.random(e - s) < mag).astype(int)
            moves[0] = 1  # the intruder is seen at least once at the start
            readings[cfg.MOTION][s:e] = moves
        elif atype in ("door_open_night", "door_left_open"):
            open_door(readings[cfg.DOOR], readings[cfg.MOTION], s, e - s, rng, n_home)


def heating_fall_rate(anomalies, n):
    """Per-minute fall rate (degC/hour) for heating failures, NaN elsewhere."""
    rate = np.full(n, np.nan)
    for a in anomalies:
        if a["anomaly_type"] == "heating_failure":
            rate[a["start"]:a["end"]] = a["magnitude"]
    return rate


def extend_heating_labels(anomalies, temp_failed, temp_normal):
    """Extend each heating-failure label through the recovery.

    The label ends at the first minute where the noise-free series is back within
    `recovery_tolerance` of the no-failure series (capped at the reserved recovery time).
    """
    spec = cfg.ANOMALIES["heating_failure"]
    for a in anomalies:
        if a["anomaly_type"] != "heating_failure":
            continue
        e = a["end"]
        gap = np.abs(temp_failed[e:e + spec["recovery_reserve_minutes"]]
                     - temp_normal[e:e + spec["recovery_reserve_minutes"]])
        recovered = np.flatnonzero(gap <= spec["recovery_tolerance"])
        if len(recovered) == 0:
            print(f"  warning: heating failure at minute {a['start']} not recovered within reserve")
            a["end"] = e + spec["recovery_reserve_minutes"]
        else:
            a["end"] = e + int(recovered[0])


# ============================================================
# 6. Data-quality issues
# ============================================================
def apply_sensor_outages(readings, outages):
    """Set a sensor's readings to NaN for the outage span."""
    for o in outages:
        if o["kind"] == "sensor_outage":
            readings[o["sensor_id"]][o["start"]:o["end"]] = np.nan


def apply_dropped_readings(readings, protected, rng):
    """Single lost messages, never inside a labelled anomaly."""
    rate = cfg.DATA_QUALITY["dropped_reading_rate"]
    for sensor in cfg.SENSORS:
        drop = (rng.random(len(protected)) < rate) & ~protected
        readings[sensor][drop] = np.nan


def build_frame(index, readings):
    """Wide readings table with rounded values; binary sensors as nullable ints."""
    df = pd.DataFrame({"timestamp": index})
    df[cfg.TEMP] = np.round(readings[cfg.TEMP], 2)
    df[cfg.POWER] = np.round(readings[cfg.POWER], 1)
    df[cfg.MOTION] = pd.array(np.round(readings[cfg.MOTION]), dtype="Int64")
    df[cfg.DOOR] = pd.array(np.round(readings[cfg.DOOR]), dtype="Int64")
    return df


def apply_hub_outages_and_duplicates(df, outages, bounds, rng):
    """Remove rows during hub outages; duplicate a few rows per split."""
    hub_rows = np.zeros(len(df), dtype=bool)
    for o in outages:
        if o["kind"] == "hub_outage":
            hub_rows[o["start"]:o["end"]] = True

    duplicates = []
    for split, (lo, hi) in bounds.items():
        candidates = np.flatnonzero(~hub_rows[lo:hi]) + lo
        for pos in rng.choice(candidates, size=cfg.DATA_QUALITY["duplicate_rows_per_split"], replace=False):
            duplicates.append({"kind": "duplicate_row", "sensor_id": "all",
                               "start": int(pos), "end": int(pos) + 1, "split": split})

    dup_rows = df.iloc[[d["start"] for d in duplicates]]
    out = pd.concat([df[~hub_rows], dup_rows]).sort_values("timestamp", kind="stable")
    return out.reset_index(drop=True), duplicates


# ============================================================
# 7. Save outputs
# ============================================================
def spans_to_frame(spans, index):
    """Convert minute positions to timestamps; `end` is the last affected minute (inclusive)."""
    df = pd.DataFrame(spans)
    df["start"] = index[df["start"]]
    df["end"] = index[df["end"] - 1]
    return df


def save_outputs(df, anomalies, log_items, index):
    """Write per-split readings, anomaly labels and the generation log."""
    cfg.DATA_DIR.mkdir(exist_ok=True)
    for split, (s, e) in cfg.SPLITS.items():
        part = df[(df["timestamp"] >= s) & (df["timestamp"] < e)]
        part.to_csv(cfg.SPLIT_FILES[split], index=False, date_format=cfg.TIMESTAMP_FORMAT)

    labels = spans_to_frame(anomalies, index).sort_values("start").reset_index(drop=True)
    labels["magnitude"] = labels["magnitude"].round(2)
    labels.insert(0, "anomaly_id", [f"A{i + 1:03d}" for i in range(len(labels))])
    labels = labels[["anomaly_id", "anomaly_type", "sensor_id", "start", "end",
                     "magnitude", "magnitude_unit", "split"]]
    labels.to_csv(cfg.LABELS_FILE, index=False, date_format=cfg.TIMESTAMP_FORMAT)

    log = spans_to_frame(log_items, index).sort_values("start").reset_index(drop=True)
    log = log[["kind", "sensor_id", "start", "end", "split"]]
    log.to_csv(cfg.GENERATION_LOG_FILE, index=False, date_format=cfg.TIMESTAMP_FORMAT)
    return labels, log


def print_summary(df, labels, log):
    """Row counts, anomaly counts and anomalous-minute share per split and sensor."""
    print("\nRows per split:")
    for split, (s, e) in cfg.SPLITS.items():
        print(f"  {split:5s}: {((df['timestamp'] >= s) & (df['timestamp'] < e)).sum():,}")

    print("\nAnomalies per type and split:")
    print(labels.pivot_table(index="anomaly_type", columns="split", values="anomaly_id",
                             aggfunc="count")[list(cfg.SPLITS)].to_string())

    print("\nAnomalous minutes (% of split minutes, per sensor):")
    minutes = (labels["end"] - labels["start"]).dt.total_seconds() / 60 + 1
    rows = []
    for split, (s, e) in cfg.SPLITS.items():
        split_minutes = (pd.Timestamp(e) - pd.Timestamp(s)).total_seconds() / 60
        row = {"split": split}
        for sensor in cfg.SENSORS:
            mask = (labels["split"] == split) & (labels["sensor_id"] == sensor)
            row[sensor] = round(100 * minutes[mask].sum() / split_minutes, 2)
        row["any_sensor"] = round(100 * minutes[labels["split"] == split].sum() / split_minutes, 2)
        rows.append(row)
    print(pd.DataFrame(rows).set_index("split").to_string())

    print("\nGeneration log (benign events + data quality):")
    print(log.pivot_table(index="kind", columns="split", values="start",
                          aggfunc="count")[list(cfg.SPLITS)].fillna(0).astype(int).to_string())


# ============================================================
# Main
# ============================================================
def main():
    rng = np.random.default_rng(cfg.SEED)
    index = build_index()
    bounds = split_bounds(index)
    tod = np.arange(len(index)) % MIN_PER_DAY

    # Occupancy
    occupancy, trips = simulate_occupancy(index, rng)
    door_trips = add_extra_door_trips(occupancy, trips, rng)
    n_home = occupancy["n_home"].to_numpy()
    target = thermostat_target(occupancy)

    # Plan injections (outages -> benign -> anomalies)
    planner = SpanPlanner(rng)
    outages = plan_outages(planner, bounds, len(index))
    benign = plan_benign_events(planner, index, bounds, occupancy)
    masks = anomaly_allowed_masks(index, occupancy, target, door_trips)
    anomalies = plan_anomalies(planner, bounds, masks)

    # Temperature: noise-free with and without failures, then label the recovery
    solar = solar_gain(tod)
    temp_failed = temperature_physics(target, solar, heating_fall_rate(anomalies, len(index)))
    temp_normal = temperature_physics(target, solar, np.full(len(index), np.nan))
    extend_heating_labels(anomalies, temp_failed, temp_normal)

    # Normal behaviour for the other sensors
    motion, door = simulate_motion_and_door(tod, occupancy, door_trips, rng)
    readings = {
        cfg.TEMP: add_sensor_noise(temp_failed, cfg.TEMPERATURE["noise_sd"], rng),
        cfg.POWER: simulate_fridge(tod, n_home, rng),
        cfg.MOTION: motion.astype(float),
        cfg.DOOR: door.astype(float),
    }

    # Overlays
    apply_benign_events(readings, benign, n_home, rng)
    apply_anomalies(readings, anomalies, n_home, rng)

    # Data quality
    protected = np.zeros(len(index), dtype=bool)
    for a in anomalies:
        protected[a["start"]:a["end"]] = True
    apply_sensor_outages(readings, outages)
    apply_dropped_readings(readings, protected, rng)
    df = build_frame(index, readings)
    df, duplicates = apply_hub_outages_and_duplicates(df, outages, bounds, rng)

    # Save
    labels, log = save_outputs(df, anomalies, benign + outages + duplicates, index)
    print_summary(df, labels, log)


if __name__ == "__main__":
    main()
