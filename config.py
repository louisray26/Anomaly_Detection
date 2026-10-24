"""Central configuration: sensors, dates, splits, seed, paths and generator parameters.

All values here mirror the decisions recorded in docs/design.md (Stage 3).
"""

from pathlib import Path

# ============================================================
# Reproducibility
# ============================================================
SEED = 42

# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SPLIT_FILES = {
    "train": DATA_DIR / "train.csv",
    "val": DATA_DIR / "val.csv",
    "test": DATA_DIR / "test.csv",
}
LABELS_FILE = DATA_DIR / "anomaly_labels.csv"
GENERATION_LOG_FILE = DATA_DIR / "generation_log.csv"

# ============================================================
# Timeline and splits
# ============================================================
START = "2024-01-08 00:00"  # Monday
N_DAYS = 28
FREQ = "1min"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"

# Inclusive start, exclusive end
SPLITS = {
    "train": ("2024-01-08", "2024-01-22"),  # days 1-14
    "val": ("2024-01-22", "2024-01-29"),    # days 15-21
    "test": ("2024-01-29", "2024-02-05"),   # days 22-28
}

# ============================================================
# Sensors
# ============================================================
TEMP = "temp_living_room"
POWER = "power_fridge"
MOTION = "motion_hallway"
DOOR = "door_front"
SENSORS = [TEMP, POWER, MOTION, DOOR]

# ============================================================
# Occupancy (times as minutes after midnight)
# ============================================================
OCCUPANCY = {
    "wake_mean": 6 * 60 + 30,
    "sleep_mean": 23 * 60,
    "jitter_sd": 15,                         # minutes
    "jitter_max": 30,                        # clip to +/- this
    "leave_window": (8 * 60, 8 * 60 + 45),
    "return_window": (17 * 60, 18 * 60 + 30),
    "office_days_a": [0, 1, 2, 3, 4],        # adult A: office Mon-Fri
    "office_days_b": [1, 2, 3],              # adult B: hybrid, office Tue-Thu
    "weekend_outing_window": (10 * 60, 17 * 60),
    "weekend_outing_hours": (2, 4),          # adult A only; adult B stays home
    "extra_door_openings": (0, 2),           # per day, when someone is home
    "door_open_minutes": (1, 3),
    "motion_around_door": (1, 3),            # minutes of motion before/after
}

# ============================================================
# Normal sensor behaviour
# ============================================================
TEMPERATURE = {
    "target_home": 21.0,
    "target_night": 18.0,
    "target_eco": 18.0,                      # house empty
    "lag_heat_minutes": 45,                  # time constant when heating up
    "lag_cool_minutes": 300,                 # time constant when cooling down (slower)
    "solar_gain": 0.5,                       # peak extra degC
    "solar_window": (12 * 60, 16 * 60),
    "noise_sd": 0.1,
}

FRIDGE = {
    "on_power": (100, 130),
    "off_power": (2, 5),
    "on_minutes": (12, 18),
    "off_minutes": (18, 30),
    "noise_sd_on": 2.0,                      # compressor running / defrost
    "noise_sd_off": 0.3,                     # idle (keeps idle clearly above 0 W)
    "meal_on_factor": 1.2,                   # longer runs at mealtimes
    "meal_windows": [(7 * 60, 8 * 60 + 30), (12 * 60, 13 * 60 + 30), (18 * 60, 20 * 60 + 30)],
    "defrost_every_hours": (9.5, 10.5),
    "defrost_minutes": 20,
    "defrost_power": (280, 320),
}

MOTION_PROB = {
    "asleep": 0.005,
    "away": 0.001,                           # stray sensor triggers in an empty house
    "home_base": 0.10,
    "home_morning": 0.30,
    "home_evening": 0.25,
    "morning_window": (6 * 60 + 30, 8 * 60 + 30),
    "evening_window": (17 * 60 + 30, 22 * 60),
}

# ============================================================
# Injected anomalies (labelled)
# ============================================================
ANOMALY_COUNTS = {"train": 3, "val": 2, "test": 2}   # per type
ANOMALY_GAP_MINUTES = 120        # min gap between two anomalies
RESERVED_BUFFER_MINUTES = 30     # min gap to benign events / outages

ANOMALIES = {
    "temp_spike": {
        "sensor": TEMP, "minutes": (1, 3), "magnitude": (3.0, 8.0), "unit": "degC"},
    "heating_failure": {
        "sensor": TEMP, "hours": (1.5, 3), "magnitude": (0.5, 1.0), "unit": "degC_per_hour",
        "min_total_drop": 1.5,
        "recovery_tolerance": 0.3,           # label ends when within this of the no-failure series
        "recovery_reserve_minutes": 150},    # kept free after the failure for the recovery
    "fridge_power_loss": {
        "sensor": POWER, "hours": (0.75, 2), "magnitude": (0.0, 0.5), "unit": "W"},
    "compressor_stuck_on": {
        "sensor": POWER, "hours": (1, 2), "magnitude": (100, 130), "unit": "W"},
    "intrusion_motion": {
        "sensor": MOTION, "minutes": (5, 30), "magnitude": (0.3, 0.6), "unit": "fraction_of_minutes",
        "empty_margin_minutes": 15},
    "door_open_night": {
        "sensor": DOOR, "minutes": (1, 5), "unit": "min", "window": (30, 5 * 60)},
    "door_left_open": {
        "sensor": DOOR, "minutes": (20, 60), "unit": "min"},
}

# ============================================================
# Benign rare events (unlabelled, logged only)
# ============================================================
BENIGN = {
    "night_bathroom_trip": {"per_week": 3, "window": (60, 5 * 60), "minutes": (1, 4)},
    "weekend_guests": {"per_split": 1, "window": (19 * 60, 23 * 60 + 30), "hours": (3, 4),
                       "door_openings": (3, 6), "motion_prob": (0.3, 0.5)},
}

# ============================================================
# Data quality issues (not anomalies, logged only)
# ============================================================
DATA_QUALITY = {
    "sensor_outage_minutes": (10, 120),      # 1 per sensor per split
    "hub_outage_minutes": (20, 60),          # 1 per split, rows removed
    "duplicate_rows_per_split": 2,
    "dropped_reading_rate": 0.001,           # per sensor, single NaN cells
}

# ============================================================
# Detection (Stage 5) - see docs/design.md
# ============================================================
ARTIFACTS_DIR = ROOT / "artifacts"
OUTPUTS_DIR = ROOT / "outputs"

# User-configured schedule: used ONLY by the rule baseline, never by ML models
SCHEDULE = {
    "away": {"days": [1, 2, 3], "window": (9 * 60, 17 * 60)},   # Tue-Thu 09:00-17:00
    "night": {"window": (0, 5 * 60)},                           # 00:00-05:00 every day
}
# Sensitivity check only (not used for selection): wider windows that no longer match the injection windows
SCHEDULE_SENSITIVITY = {
    "away": {"days": [1, 2, 3], "window": (8 * 60, 18 * 60 + 30)},  # Tue-Thu 08:00-18:30
    "night": {"window": (23 * 60 + 30, 6 * 60)},                  # 23:30-06:00 (wraps midnight)
}

# Fixed rule thresholds (physics / house rules / unlabelled-train valleys)
RULES = {
    "spike_resid_degc": 1.5,
    "heating_sustain_minutes": 15,
    "power_loss_watts": 1.0,
    "power_loss_minutes": 10,
    "stuck_on_minutes": 30,
    "intrusion_min_motion_15": 2,
    "door_left_open_minutes": 10,
}

# Tuned on val (listed from least to most conservative)
RULE_GRID = {
    "heating_dev_degc": [0.5, 0.75, 1.0],
    "heating_rate_degc_h": [0.2, 0.3, 0.4],
    "intrusion_no_door_minutes": [15, 30, 60],
}

IFOREST = {"n_estimators": 200}
IF_PERCENTILE_GRID = [99.0, 99.5, 99.9, 99.95, 99.99]   # of train scores
IFOREST_V2 = {"n_estimators": 200, "max_samples": 1024}  # fewer, sensor-specific inputs

# Control charts: fixed textbook constants; only the limit (and chart type) is tuned on val
CHARTS = {"ewma_lambda": 0.2, "cusum_k": 0.5}
CHART_GRID = {"ewma": [3, 4, 5, 6, 8], "cusum": [5, 10, 20, 40, 80]}   # least -> most conservative

# Alerts and scoring
ALERT_MERGE_GAP_MINUTES = 10
GRACE_MINUTES = 15
