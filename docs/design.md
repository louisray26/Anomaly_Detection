# Design Decisions

## Stage 1 – Problem Understanding

### Setting
- Real-time anomaly detection for a single DIY smart home, running **on-device** (Raspberry Pi / Home Assistant class hardware).
- Cloud retraining and model updates are covered only in the README's productionization section.
- No real-world labels exist. Detection must be unsupervised or rule-based. Injected labels are used only for evaluation.
- Real-time means **causal**: each decision uses only past and current readings (no centered windows or look-ahead).
- False alarms and missed events are treated as **equally costly**. The exact success metric will be defined in the evaluation stage.
- Alerts must be explainable to a non-expert user.

### Terminology
This document keeps the two terms strictly separate:

- **Event:** normal activity. It is never labelled and never counted as a detection target.
  - **Routine events:** everyday activity (door opens when leaving, evening motion, compressor cycle, defrost).
  - **Benign rare events:** unusual but normal activity (night bathroom trip, weekend guests). They test the false-alarm rate.
- **Anomaly:** an observation that is unlikely given its context (time of day, day of week, duration, or consistency with other sensors). Anomalies are injected, labelled, and are the detection targets.

The same reading can be an event or an anomaly depending on context (e.g. door opening at 08:15 vs. 03:10), so value thresholds alone are not enough.

### Sensors

| Sensor | Data type | Normal behaviour |
|---|---|---|
| Indoor temperature | Continuous, slow | 19–23 °C. Heating schedule (warm-up ~06:30, setback ~23:00), slight afternoon solar gain, noise ~±0.1 °C |
| Fridge power | Continuous, cyclic | Compressor ~80–150 W for 10–20 min, idle ~1–5 W, duty cycle 30–50%, defrost every 8–12 h, brief bumps on door opening |
| Hallway motion | Binary per minute | Follows occupancy: morning/evening peaks, quiet when away, sparse at night |
| Front door | Binary (0 closed / 1 open) | Mostly closed. 1–3 min openings when leaving or returning |

### Household and season assumptions
- Two adults. One works in the office every weekday. The other works hybrid: **at home Mon and Fri, office Tue–Thu**.
- The house is empty during office hours on **Tue–Thu only**. Mon/Fri daytime and weekends are occupied.
- WFH days are fixed rather than random, so "house empty" is learnable from the weekly pattern and the intrusion anomaly stays well defined.
- **Winter heating season**, with no daylight-saving change during the 28 days.

### Injected anomalies (7 types, labelled)

| # | Anomaly | Sensor | Category | Meaning |
|---|---|---|---|---|
| 1 | Temperature spike | temp | point | Sensor glitch |
| 2 | Heating failure (slow drift down) | temp | collective | Boiler fault |
| 3 | Power drop to ~0 W for hours | fridge | collective | Failure / unplugged |
| 4 | Compressor never stops | fridge | collective | Door ajar / refrigerant leak |
| 5 | Daytime motion on office day, no door entry | motion (+door) | contextual / cross-sensor | Intrusion |
| 6 | Door opened at night | door | contextual | Security breach |
| 7 | Door left open for a long time | door | collective | Forgot to close |

- Timing, magnitude and duration are randomised per event. Exact ranges are set in the dataset stage.
- **Counts:**
  - Train: at least 3 per type (≥21 anomalies). Present in the data, but labels are never used as model input (realistic contamination).
  - Validation and test: exactly 2 per type (14 anomalies each), so per-type recall is not limited to 0% or 100%.
- **Keep the anomaly rate small.** Check the % of anomalous minutes (overall and per sensor) after generation. If it is too high, shorten the long-duration anomalies.

### Injected benign rare events (unlabelled distractors)
- **Night bathroom trip:** brief hallway motion around 02:00–04:00, no door activity.
- **Weekend guests:** extra door openings and motion on a Saturday evening.

### Data quality
- Sensor outages and missing observations are a **data-quality issue**, neither an event nor an anomaly.
- They are handled during cleaning and reported separately, and excluded from anomaly metrics.

---

## Stage 2 – Repo Structure

```
├── README.md
├── requirements.txt
├── config.py             # sensors, split dates, seeds, paths
├── generate_data.py      # synthetic data + labels → data/
├── features.py           # shared cleaning + causal features
├── analysis.ipynb        # cleaning decisions, EDA plots
├── detect_anomalies.py   # train/compare models on train+val, save best, write alerts
├── inference.py          # stream test days line by line with saved model
├── data/                 # sensor readings CSV and labels CSV (separate files)
├── artifacts/            # saved model, thresholds, preprocessing config
├── outputs/              # alerts CSVs, metrics
├── docs/                 # plan, task, design
└── ai_transcripts/       # exported AI session transcripts, one per phase
```

### Decisions
- **`features.py`:** single source of cleaning and feature logic, shared by `detect_anomalies.py` and `inference.py`, so training and production cannot drift apart. `analysis.ipynb` justifies the choices but does not hold production code.
- **`config.py`:** single place for sensors, split dates, random seed and paths.
- **`detect_anomalies.py`:** writes an alerts CSV (timestamp, sensor, score/rule, reason), as the task requires, in addition to comparing and saving models.
- **Generated data is committed to git.** Sensor readings and labels are kept in separate files. The generator is fully reproducible from the seed in `config.py`.

---

## Stage 3 – Dataset Design, Creation and Split

### Format and identifiers
- **Sensor IDs:** `temp_living_room`, `power_fridge`, `motion_hallway`, `door_front`.
- **Dates:** Mon 2024-01-08 to Sun 2024-02-04, 1-minute sampling, local time with no time zone (winter, no daylight-saving change). Splits align with whole Mon–Sun weeks.
- **`data/train.csv`, `data/val.csv`, `data/test.csv`:** wide format, one row per minute: `timestamp, temp_living_room, power_fridge, motion_hallway, door_front`. Missing readings are NaN. Chosen because line-by-line streaming gets a full home snapshot per row.
- **`data/anomaly_labels.csv`:** one row per injected anomaly: `anomaly_id, anomaly_type, sensor_id, start, end, magnitude, magnitude_unit, split`. Per-minute labels are derived from this table. `magnitude_unit` is needed because magnitude means different things per type (°C, W, minutes, …).
- **`data/generation_log.csv`:** benign events and outages: `kind, sensor_id, start, end`. Never used for scoring. It is only for diagnosing false alarms and verifying cleaning.

### Normal behaviour
- **Occupancy (drives all sensors):**
  - Wake ~06:30, sleep ~23:00 (±30 min daily jitter).
  - **Tue–Thu:** both adults out ~08:00–08:45 → ~17:00–18:30, so the house is empty in between.
  - **Mon/Fri:** one adult stays home.
  - **Weekends:** one adult always stays home. The other has one 2–4 h outing per day. The house is never empty at weekends.
  - So "house empty" only occurs on Tue–Thu daytime.
- **Indoor temperature:**
  - Thermostat targets 21 °C while awake and home, 18 °C at night, and 18 °C eco mode while the house is empty (Tue–Thu daytime).
  - Room temperature follows the target with a first-order lag: **45 min when heating, 300 min when cooling**. Normal setback cooling (23:00 night setback, Tue–Thu eco mode) starts at ~0.6 °C/h and slows to ~0.2 °C/h. It overlaps the slowest heating failures (0.5 °C/h), so a failure is only identifiable in context: falling while the target is 21 °C.
  - Afternoon solar gain of ~+0.5 °C, sensor noise ~0.1 °C.
- **Fridge power:**
  - Compressor 100–130 W for 12–18 min, idle 2–5 W for 18–30 min (~40% duty).
  - Noise depends on state: ±2 W when running or defrosting, ±0.3 W at idle. Idle never reads near 0 W (measured minimum 1.1 W), so it cannot be confused with a power loss.
  - Defrost every ~10 h, ~20 min at ~300 W (normal, but the highest reading).
  - Runs ~20% longer at mealtimes when someone is home.
- **Hallway motion:**
  - Per-minute probability: ~0.10–0.30 awake and home (morning/evening peaks), ~0.005 asleep, **0.001 when away** (stray sensor triggers, ~0.5 per empty day).
  - Motion in the **first and last minute** of every door opening (walk through, close), not throughout. Motion **before** opening only if someone was already inside (never on arrival to an empty house). Motion after closing only if someone is inside.
- **Front door:**
  - Opens 1–3 min at each departure or arrival.
  - 0–2 extra openings per day when someone is home.

### Anomaly injection
Random start within the allowed window, with random duration and magnitude per anomaly. Ranges are provisional and will be revisited after generation, once the anomalous-minute rate is measured.

| Type | When allowed | Duration | Magnitude (unit) |
|---|---|---|---|
| `temp_spike` | Any time | 1–3 min | ±3–8 (°C jump, random sign) |
| `heating_failure` | 21 °C target throughout the failure **and** the 150-min recovery reserve | 1.5–3 h fall + recovery | Fall rate 0.5–1.0 °C/h, **total drop ≥ 1.5 °C**. Label covers the recovery too |
| `fridge_power_loss` | Any time | 45–120 min | Power reads 0–0.5 W |
| `compressor_stuck_on` | Any time | 1–2 h | Constant 100–130 W (normal run ≤ 18 min) |
| `intrusion_motion` | Tue–Thu while the house is empty | 5–30 min | Share of minutes with motion 0.3–0.6: exactly max(2, round(share × duration)) motion minutes, first minute always on, no door opening |
| `door_open_night` | 00:30–05:00 | 1–5 min | Minutes open, with hallway motion around it |
| `door_left_open` | Awake and home | 20–60 min | Minutes open (normal 1–3 min) |

- **Heating failure ≥ 1.5 °C:** sample the fall rate first, then the duration from [max(1.5 h, 1.5 / rate), 3 h]. At the lowest rate this forces a 3 h failure.
- **Heating-failure label includes recovery:** the label ends at the first minute where the noise-free temperature is back within 0.3 °C of a noise-free run without the failure. Alerts during recovery are correct detections, not false alarms. The noise-free physics is run twice; noise is added once afterwards, so the random stream is not disturbed.
- **Duration history:**
  - The original ranges (heating 2–5 h, power loss 1–4 h, stuck-on 2–4 h, door left open 30–90 min) gave 10.4–12.6% anomalous minutes per split.
  - Shortened ranges plus the reviewer fixes (cooling lag, labelled recovery) give the following (measured, seed 42):

  | Split | Temp | Fridge | Motion | Door | Any sensor |
  |---|---|---|---|---|---|
  | Train | 3.99% | 2.48% | 0.32% | 0.54% | 7.33% |
  | Val | 4.91% | 3.46% | 0.37% | 1.10% | 9.84% |
  | Test | 5.24% | 2.56% | 0.37% | 0.99% | 9.16% |

  - Temperature rose again because the heating labels now include recovery (~4–5 h per failure).
- **Placement:**
  - No overlap between anomalies, with a gap of at least 2 h.
  - Anomalies never overlap benign events or outages, so each label stays unambiguous.

### Benign events (unlabelled, logged in `generation_log.csv`)

| Kind | When | What happens | How often |
|---|---|---|---|
| `night_bathroom_trip` | 01:00–05:00 | Motion for 1–4 min, no door opening | ~3 per week |
| `weekend_guests` | Saturday 19:00–23:30 | 3–6 extra door openings, motion 0.3–0.5 for 3–4 h | Once per split (one Saturday each in train, val, test) |

- **Night bathroom trip:** the hard negative for `door_open_night` (motion without the door).
- **Weekend guests:** the hard negative for `door_left_open` and intrusion (a busy but normal evening).
- Guests are not every Saturday, so they stay rare rather than becoming a weekly routine.

### Missing data and outages (data quality, not anomalies)
Placed **per split**, so train, val and test all contain the same kinds of data-quality issues and the test stream must handle them.

| Kind | Simulates | Appears in CSV as | Per split | Total |
|---|---|---|---|---|
| `sensor_outage` | One sensor offline | That column is NaN; rows present | 1 per sensor, 10–120 min | 12 |
| `hub_outage` | Whole hub offline | Rows missing (timestamp gap) | 1, 20–60 min | 3 |
| `duplicate_row` | Hub re-sends a message | Same timestamp twice | ~2 | ~6 |
| `dropped_reading` | Single lost message | One NaN cell | ~0.1% random | – |

- Everything except `dropped_reading` is recorded in `generation_log.csv` so cleaning can be verified.

### Split

| Split | Days | Dates | Purpose |
|---|---|---|---|
| Train | 1–14 | Mon 2024-01-08 – Sun 2024-01-21 | Fit models / thresholds |
| Validation | 15–21 | Mon 2024-01-22 – Sun 2024-01-28 | Threshold tuning and model selection |
| Test | 22–28 | Mon 2024-01-29 – Sun 2024-02-04 | Run once at the end |

- One continuous 28-day series is simulated and then cut at the split dates. Sensor state carries smoothly across boundaries.
- No anomaly, benign event or outage crosses a split boundary.
- **Three readings files** (`train.csv`, `val.csv`, `test.csv`). The test file is the input for streaming, and keeping it separate makes accidental use during tuning harder.
- One `anomaly_labels.csv` (with a `split` column) and one `generation_log.csv`.

---

## Stage 4 – Cleaning & EDA

### Data scope (leakage rules)

| EDA part | Data used |
|---|---|
| Sensor behaviour (time series, daily/weekly profiles, distributions) | Train only |
| Data-quality checks (missing values, duplicates, gaps, timestamp consistency) | All splits (mechanical, same as production) |
| Anomaly counts / affected minutes per type and sensor | All splits, from `anomaly_labels.csv` (documentation only) |
| Plots of injected anomalies | Train only |

- Val and test readings are not visually inspected before final evaluation.
- **Any parameter learned from data (e.g. scaling statistics) is fit on train only** and then applied unchanged to val/test and the production stream. Nothing is ever re-fit on val/test.

### Cleaning policy
Every step is causal (past data only), so the same code serves batch and streaming.

| Step | Check | Action |
|---|---|---|
| 1 | Timestamp order | Sort; in a stream, ignore rows older than the last seen |
| 2 | Duplicate timestamps | Keep first |
| 3 | Gaps (hub outage) | Reindex to a regular 1-min grid; missing rows become NaN |
| 4 | Physically impossible values | Set to NaN, using fixed hardware limits (not learned) |
| 5 | Short gaps | Forward-fill, **at most 2 min** (all sensors) |
| 6 | Longer gaps | Leave NaN, flag sensor **offline**, skip detection for that sensor. Reported as data quality, not anomalies |

- **Outliers are not clipped or removed in cleaning.** They are the detection targets (e.g. `temp_spike`). Learned limits belong to the detectors.
- No interpolation, because it would use future values.

### EDA plots (train only; train labels used for shading and normal-vs-anomalous splits)
1. One train week, 4 stacked sensor panels, anomalies shaded by type.
2. Daily profile by day type (Tue–Thu / Mon & Fri / weekend).
3. Distributions, normal vs anomalous minutes, per sensor.
4. Fridge run lengths per mode, normal vs anomalous.

Plus tables: a data-quality summary per split, and anomaly counts / affected minutes by type and sensor.

### Preprocessing (implemented in `features.py`, all causal)

| Sensor | Features |
|---|---|
| Temperature | **Deviation from the train baseline** (`temp_dev`, median for this hour × baseline group). **Residual from recent readings** (`temp_resid` = reading − 15-min rolling median). Rate of change over 15 and 60 min (°C/h), computed on the **15-min rolling median** |
| Fridge | Mode from **fixed thresholds**: idle < 50 W, running 50–200 W, defrost ≥ 200 W (not tuned). Minutes in current mode |
| Motion | Motion minutes in the last 15 and 60 min |
| Door | Minutes the door has been open. **Minutes since the door was last open** (capped at 180 min) |
| All | Hour of day, day type (`tue_thu`, `mon_fri`, `weekend`) |

- **Temperature baseline:** the thermostat target (21 °C) is not observed by any sensor, so the expected level is learned. It is the median temperature per hour × **baseline group** on **train only**, fit once and then applied unchanged to val/test.
  - **Baseline groups:** `tue_thu` (eco mode) vs `occupied` (Mon/Fri + weekend). The three day types are kept for the other features.
  - **Why pooled:** with 3 day types, train has only 4 weekend days, and 2 of them have heating failures. Up to **50%** of the minutes in some weekend-evening medians were failure minutes, which pulled the weekend 18–19 h baseline ~0.2 °C low. Pooling gives 8 days per hour; the worst cell is now **25%** contaminated (occupied, 18 h). The occupied 18–19 h baseline is 21.17 / 21.14 °C.
  - It uses the median **without labels**, because a real system has no labels.
  - This is the only learned preprocessing parameter.
- **Spike residual (`temp_resid`):** `temp_dev` alone can hide a weak spike (e.g. +3 °C during a warm-up lag at −1.8 °C lands inside the normal range). The residual from the 15-min rolling median is independent of context.
  - Normal train: std 0.12 °C, range −0.46 to +0.60 °C.
  - Train spikes: −4.11, +3.62, +3.75 / +3.57 °C.
- **Cross-sensor door link (`minutes_since_door_open`):** intrusion is defined as motion without door entry, so motion needs a link to door activity.
  - **Cross-sensor features are unknown (NaN) when a sensor they depend on was offline since the last event; the detector skips them.** For example, a return home during a door outage would otherwise look like motion with no entry, i.e. an intrusion.
  - Capped at 180 min (= warm-up length), so the value never depends on how much history is loaded. The reviewer suggested NaN until the first opening; the cap keeps warm-up and continuous runs identical.
  - Train intrusion motion minutes: 95–180 min since the door was last open.
  - Normal motion in Tue–Thu 09–17 h: 28 of 49 minutes are ≤ 5 min after a door opening (returns home). 5 minutes are > 30 min after one (3 stray triggers, 2 at 17:58–17:59 after an earlier return).
- **Label-split statistics are descriptive only.** Detector thresholds come from physics, or from unlabelled train distributions (e.g. high quantiles), and are tuned on val. The label-split numbers in the EDA findings (e.g. idle ≤ 30 vs 72–137 min) must not be copied into thresholds.
- **Fridge mode thresholds** are fixed from the physics (idle ≤ ~5 W, compressor 100–130 W, defrost ~300 W), with no tuning. Power loss appears as idle mode with an unusually long run and a near-0 W value.
- **No scaling:** the rules use physical units and Isolation Forest is scale-invariant. If a later model needs scaling, it is fit on train only.
- **Smoothed temperature rate:** a single-reading difference over 15 min is dominated by sensor noise (std 0.56 °C/h on stable normal minutes). Comparing 15-min rolling **medians** gives a std of 0.18 °C/h.
  - A rolling mean was slightly smoother (0.14 °C/h), but a spike leaked into the rate for up to 30 min afterwards (normal-minute minimum −2.51 °C/h). The median ignores 1–3 min spikes (minimum −1.24 °C/h).
  - With the median, `temp_rate_60` during a heating failure reads close to the true fall rate (−0.62 to −0.66 vs 0.65–0.69 °C/h).
- **Split-start warm-up:** val/test features are built with the last 180 min of the previous split prepended (`split_features`), then trimmed.
  - Verified: the first hour of val and test is identical to one continuous run over all 28 days (all 15 numeric features). Without warm-up it has 93 NaN and 115–128 wrong feature cells.
  - The warm-up rows are past data (a live stream has them) and are never used to fit anything.
  - Train has no earlier split, so its first hour stays without history.
- **Generator start-up fix:** the temperature simulation now starts from a settled midnight value (a burn-in run of day 1's schedule) instead of the 18 °C setback. Mon 8 Jan 00–07 h now averages 19.33 °C vs 19.36 °C on other nights (was ~18 °C). It uses no random numbers, so all other data and every label are unchanged.

All feature statistics quoted above (residual, door link, baseline contamination, rate noise, warm-up) are reproduced in `analysis.ipynb`, section 5 "Feature checks".

### EDA findings (from `analysis.ipynb`, plots in `outputs/eda/`)
- **Data quality:** per split there are 2 duplicate rows, 1 hub outage (26 / 40 / 33 missing rows) and 0 out-of-order timestamps. Cleaning detects exactly 2 offline periods per sensor per split, matching the generation log. Offline minutes = logged minutes − 2 per outage (the forward fill).

| Sensor | Observation (train) | Implication for detection |
|---|---|---|
| Temperature | Tue–Thu daytime ~2 °C lower (eco mode). Train spikes at −4.3 / +3.5 to +3.7 °C deviation vs normal −1.8 to +1.5 °C. Heating failure + recovery spans −2.1 to +0.1 °C, overlapping normal | `temp_resid` catches spikes in any context. Heating failure needs duration / rate in context |
| Motion (cross-sensor) | Intrusions happen 95–180 min after the last door opening; most normal empty-hours motion is ≤ 5 min after one | `minutes_since_door_open` + expected-empty context |
| Fridge | No daily pattern. Stuck-on power equals normal running. Power loss ≤ 0.6 W vs idle ≥ 1.1 W | **Mode duration** separates: normal idle ≤ 30, running ≤ 22 min vs anomalous 72–137 / 77–93 min |
| Motion | ~0 Tue–Thu 09–16 h. Intrusion counts overlap normal home activity | Needs context (expected empty house) |
| Door | Normal openings ≤ 4 min, `door_left_open` up to 41 min. Night openings have normal duration | Duration for left-open; time of day for night |

---

## README Carry-Over
Items to include in README.md when it is written at the end:

- **Overview:** on-device, causal, real-time detection for one household. Labels are used only for evaluation.
- **Terminology:** events (routine + benign rare) vs. anomalies.
- **Anomaly definitions:** the 7 injected anomaly types with their real-world meaning.
- **Assumptions:** household profile (two adults, hybrid WFH Mon/Fri, house empty only Tue–Thu daytime, one adult always home at weekends), thermostat with night setback and eco mode, winter heating season, no daylight-saving change.
- **Data quality:** outages and missing data are handled in cleaning, not scored as anomalies.
- **Limitations:** with 2 anomalies per type in validation/test, per-type recall can only be 0%, 50% or 100%. Training contains unlabelled anomalies (contamination). Anomalous minutes are 7–10% per split, higher than a real home, mainly because heating-failure labels include the recovery.
- **Synthetic data realism:** two reviewer passes fixed:
  - fridge idle noise
  - cooling lag
  - durations
  - motion before arrival
  - stray motion when empty
  - heating recovery labels
  - door motion only at the first and last minute
  - intrusions always have ≥2 motion minutes

  Mention this as part of the process.
- **Cleaning & features:**
  - Causal throughout; only short gaps are filled (2 min); outliers are kept.
  - The train-only temperature baseline (2 groups, pooled to limit contamination) is the only learned preprocessing parameter.
  - `temp_resid` isolates spikes; `minutes_since_door_open` gives the cross-sensor intrusion link.
  - Label-split statistics are descriptive only; thresholds never come from labels.
  - Fridge modes come from fixed thresholds.
  - The temperature rate uses a rolling median.
  - Val/test get a 180-min warm-up from the previous split.
- **Heating failure is contextual:** normal setback cooling (~0.6 °C/h at its start) overlaps the slowest failures. The rate of fall alone is not enough; the detector needs the context (target 21 °C, awake and home).
- **Productionization:** on-device serving. Cloud is used for retraining and model updates. Monitoring and retraining strategy still to be written.
- **AI transcripts:** exported per phase into `ai_transcripts/`.
- **Repo layout:** the tree and a one-line role for each file (Stage 2).
- **Setup & run:** `pip install -r requirements.txt`, then the run order `generate_data.py` → `analysis.ipynb` → `detect_anomalies.py` → `inference.py`. Note that the committed data is reproducible from the seed in `config.py`.
