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

## Stage 5 – Model Comparison

### Structure: one detector per sensor (all models)
- Alerts are attributed to a sensor by construction ("correct sensor" metric).
- An offline sensor only disables its own detector.
- Alerts are explainable per sensor.
- Cross-sensor context enters as features (e.g. `minutes_since_door_open`).

| Detector | Inputs |
|---|---|
| Temperature | `temp_dev`, `temp_resid`, `temp_rate_15`, `temp_rate_60`, hour, day type |
| Fridge | `power_fridge`, `fridge_mode`, `fridge_mode_minutes` (no time context: no daily pattern in EDA) |
| Motion | `motion_15`, `motion_60`, `minutes_since_door_open`, hour, day type |
| Door | `door_open_minutes`, hour, day type |

### Rule-based baseline
Each rule outputs a flag plus a readable reason. Thresholds come from physics, house rules, or valleys in **unlabelled** train distributions, never from labels. Only d, r and D are tuned on val.

| Sensor | Rule | Condition | Threshold source |
|---|---|---|---|
| Temperature | Spike | \|`temp_resid`\| > 1.5 °C | Physics (a room can't jump 1.5 °C in minutes); normal resid ≤ 0.6 °C |
| Temperature | Heating failure | `temp_dev` < −d and `temp_rate_60` < −r, sustained 15 min | **Tuned on val** |
| Fridge | Power loss | Power < 1 W for 10 min | Valley in unlabelled train power: 0 minutes in 0.75–1.0 W, between the ~0 W cluster (269 min) and idle (≥ 1.0 W) |
| Fridge | Stuck on | Running > **30 min** | Gap in unlabelled train running periods: q99 = 22 min, none between 22 and 60 min |
| Motion | Intrusion | During configured away hours: `motion_15` ≥ 2 and no door opening in the last D min | **Tuned on val** (D). ≥ 2 ignores single stray triggers |
| Door | Left open | Open > 10 min | House rule |
| Door | Night opening | Door opens during configured night hours | Schedule |

- **Configured schedule (Option A):** away = Tue–Thu 09:00–17:00, night = 00:00–05:00, in `config.py`. It mimics a user's "away / night mode" settings.
- **Only the rule baseline gets the schedule.** The ML models (Isolation Forest, and any later model) do **not**; they must learn context from hour and day type in the data. This is a deliberate contrast between the two approaches.
- Option B (learning quiet hours from motion) was rejected: motion alone cannot tell an empty house from a sleeping one, so night bathroom trips would trigger intrusion alerts.
- The 1 W and 30 min evidence is reproduced in `analysis.ipynb` section 5 (check 5.6).
- **Circularity caveat (reviewer):** the configured schedule matches the generator's injection windows (away Tue–Thu 09–17 ≈ intrusion window; night 00–05 ≈ `door_open_night` 00:30–05:00), and `motion_15` ≥ 2 mirrors the generator's guarantee of ≥ 2 intrusion motion minutes. The same designer wrote both, so **rule recall is an upper bound**. The ML models and charts get none of this. Checked with a schedule sensitivity run (see results).

### Isolation Forest

| Aspect | Decision |
|---|---|
| Models | 4 per-sensor Isolation Forests with the Topic 1 inputs |
| Training data | Unlabelled train (contains unlabelled anomalies) |
| Encoding | Hour as sin/cos; day type and fridge mode one-hot |
| Settings | 200 trees, default `max_samples`, `random_state` = config seed; no hyperparameter tuning |
| Threshold | `contamination` is **not** used to set the cut-off. The per-sensor threshold on `score_samples` is **tuned on val**, starting from a high percentile of train scores |
| Missing inputs | Any NaN input (offline sensor, unknown `minutes_since_door_open`) → that detector skips the minute |
| Explanation | Each alert reports its most extreme input feature compared with the train distribution (e.g. "fridge_mode_minutes = 85, above 99.9% of train"). SHAP is a README extension |

### Isolation Forest v2 (one fair revision; no further iteration)
Reason: v1 inputs were diluted by 5 generic context columns.
| Aspect | Decision |
|---|---|
| Inputs | temp: `temp_dev`, `temp_resid`, `temp_rate_60` · fridge: `power_fridge`, `fridge_mode_minutes` · motion: `motion_15`, `minutes_since_door_open`, `motion_usual` · door: `door_open_minutes`, `door_usual` |
| Learned context | `motion_usual` / `door_usual` = share of active minutes per (day type, hour) on train, Laplace-smoothed (+0.5 / +1). No labels and no configured schedule are used |
| Settings | 200 trees, `max_samples` = 1024 (v1: 256); same threshold grid and tuning as v1 |

### Control charts (EWMA / CUSUM)
Reason: classic, lightweight, and less designer-biased than rules; expected to suit gradual drifts (heating failure).
| Aspect | Decision |
|---|---|
| Signals (z-scores) | temp: `temp_dev` only (lower side, median/MAD per baseline group × hour). Charts measure **shifts**; spikes are the rule's job, so `temp_resid` is not a chart signal · fridge: `power_fridge` (two-sided) and `fridge_mode_minutes` (upper), both median/MAD per fridge mode · motion / door: 0/1 reading standardised by its learned per-(day type, hour) rate, `(x − p) / √(p(1 − p))`, upper side |
| Fit on train | Centres, scales (MAD × 1.4826, with floors: temp_dev 0.1 °C, power 0.5 W, run length 1 min) and rates. Unlabelled train |
| Fixed constants | EWMA λ = 0.2 (limit in asymptotic EWMA σ); CUSUM k = 0.5 |
| Tuned on val | Chart type (EWMA or CUSUM) and limit per sensor: EWMA L ∈ {3, 4, 5, 6, 8}; CUSUM h ∈ {5, 10, 20, 40, 80}. Same objective and tie-breaks |
| Details | CUSUM restarts at 0 after a signal (without this, one heating failure kept the chart above the limit for 2.5 days). Missing minutes hold the chart state. A sensor alerts when any of its signals crosses the limit. Reason = chart + signal + side |

### Minute flags → alerts (all models)
- An alert starts at the first flagged minute, which is when it is sent in real time.
- Flagged minutes on the same sensor < 10 min apart are merged into one alert. Merging only extends an alert already sent.
- Alert record: `sensor, model, start, end, reason`, written to `outputs/alerts_<model>_<split>.csv` (the task's "detected anomalies" output).
- No minimum alert length; persistence lives inside the rules that need it. To be revisited for IF only if val shows many 1-minute false alarms.

### Scoring, tuning and selection (val)
**Scoring rules**
- **Hit:** an alert on the anomaly's sensor overlapping [anomaly start, anomaly end + **15 min grace**].
- **False alarm:** an alert on its sensor overlapping no anomaly (+ grace) on that sensor.
- **Cross-sensor alarm:** an alert on a *different* sensor during an anomaly counts as a false alarm and is also reported separately.
- **Benign-event alarms:** false alarms overlapping a benign event (from `generation_log.csv`) are reported separately, but still count as false alarms.

**Tuning (per sensor)**
- Objective: minimise **misses + false alarms** (equal cost, Stage 1).
- Tie-break 1: lower median detection delay. Tie-break 2: the more conservative setting.
- Grids: rules d ∈ {0.5, 0.75, 1.0} °C, r ∈ {0.2, 0.3, 0.4} °C/h, D ∈ {15, 30, 60} min. IF threshold at train-score percentiles {99.0, 99.5, 99.9, 99.95, 99.99}.
- Risk: only 2–4 val anomalies per sensor, so tuning can overfit. Small grids and conservative ties limit this; test gives the honest number.

**Selection**
- **One model overall**, chosen by total errors (misses + false alarms) on val across all sensors. Per-sensor results are reported alongside.

**Comparison table:** detection per sensor/type, false alarms (total, per day, cross-sensor, on benign events), median delay, model size, **inference time per minute**, number of tuned parameters, explainability.

### Val results (first run: rules vs IF v1 only; kept for history, superseded below)

| Model | Detected | False alarms | Benign FA | Cross-sensor FA | Total errors | Size | ms / minute | Tuned params |
|---|---|---|---|---|---|---|---|---|
| Rules | 14/14 | 0 | 0 | 0 | **0** | 3.2 KB | 0.0006 | 3 |
| Isolation Forest | 4/14 | 3 | 2 | 0 | 13 | 11,477 KB | 0.019 | 4 |

- **Tuned rules:** d = 0.5 °C, r = 0.2 °C/h, D = 60 min. All rule grid points tied on errors; the tie-break chose the lowest delay.
- **Tuned IF percentiles:** temp p99.95, fridge p99.9, motion p99.99, door p99.99.
- **Median delay (rules):** temp 27.5, fridge 15.5, motion 2, door 5 min.
- **IF detected only** heating failure 2/2 and power loss 2/2 (delay ~95–131 min). It missed spikes, stuck-on, intrusion, and both door types.
- **Selected: rules** (fewest total val errors).

- **Known limitation:** IF scores each minute independently. Duration is only captured where a feature encodes it (`fridge_mode_minutes`, `door_open_minutes`, `temp_rate_60`).

### Val results (all four models, seed 42; `outputs/val_*.csv`)

| Model | Detected | False alarms | Benign FA | Cross-sensor FA | Total errors | Size | ms / minute | Tuned params |
|---|---|---|---|---|---|---|---|---|
| Rules | 14/14 | 0 | 0 | 0 | **0** | 3.2 KB | 0.0004 | 3 |
| Control charts | 9/14 | 2 | 1 | 0 | 7 | 14 KB | 0.006 | 8 |
| IF v2 | 8/14 | 3 | 2 | 0 | 9 | 20,864 KB | 0.023 | 4 |
| IF v1 | 4/14 | 3 | 2 | 0 | 13 | 11,482 KB | 0.018 | 4 |

- **Tuned:** IF v2 percentiles temp p99.95, fridge p99.5, motion p99.95, door p99.5. Charts: temp EWMA 6, fridge EWMA 6, motion CUSUM 80, door CUSUM 80.
- **IF v2 vs v1:** +4 detections (temp spikes 2/2, door left open 2/2), same false alarms. Still misses stuck-on, intrusion and night door openings. Size nearly doubled (larger `max_samples`).
- **Charts per sensor:**
  - Fridge: 4/4, 0 FA (EWMA 6; delay 10 min, faster than rules' 15.5).
  - Temperature (`temp_dev` lower side only): heating failures 2/2, spikes 0/2 by design, 1 FA (Wed 01‑24 07:03–07:21, temperature below the usual 07:00 Tue–Thu level, likely a late morning heat-up). **Heating failure: charts ~21 min vs rules ~59 min** (rules need 15 sustained minutes of both dev and rate conditions; the chart accumulates evidence from the first minutes).
  - Motion: 1/2 at CUSUM 80; lower limits catch both but add 5–63 FA. 1 FA on weekend guests.
  - Door: left open 2/2, night openings 0/2. A 1–2 min night opening adds too little to the chart before it ends; limits that catch it add 3–19 FA.
- **Selected: rules** (still fewest total val errors), subject to the circularity caveat (rule recall is an upper bound).

**Median detection delay per anomaly type (min; `outputs/val_per_type.csv`)**

| Anomaly type | Rules | Charts | IF v2 | IF v1 |
|---|---|---|---|---|
| temp_spike | 0 | – (not a chart signal) | 0 | – |
| heating_failure | 59 | **21** | 86.5 | 131 |
| fridge_power_loss | 9 | **4.5** | 91 | 95 |
| compressor_stuck_on | 26 | **19** | – | – |
| intrusion_motion | **2** | 3 (1/2) | – | – |
| door_open_night | **0** | – | – | – |
| door_left_open | 10 | 4.5 | **3.5** | – |

– = not detected. Per-sensor medians mix types (e.g. rules temp 27.5 = spikes 0 + failures 59), so per-type delay is the number to read.

**Schedule sensitivity (rules only; not used for selection; `outputs/val_sensitivity.csv`)**
Only the schedule changes: away Tue–Thu 08:00–18:30, night 23:30–06:00 (`config.SCHEDULE_SENSITIVITY`); same grid and tuning (chose the same d, r, D).

| Schedule | Detected | False alarms | Total errors |
|---|---|---|---|
| Main (matches injection windows) | 14/14 | 0 | 0 |
| Wide | 14/14 | 2 (motion; 1 cross-sensor) | 2 |

- Both new FAs are 08:00 on Tue/Wed: adults are still home before leaving (08:00–08:45) and the last door opening was the evening before, so the door link does not suppress them. No new door FAs from the wider night window on val.
- Reading: rules stay best on val (2 errors vs 7 for charts), but their 0 errors depends on a schedule that fits the household exactly. In a real home the schedule is user-entered and drifts; this is a deployment risk to state in the README.

### Files produced by `detect_anomalies.py` (and why)
One run writes everything below; all are reproducible from the seed. Test is never loaded, so every file here is train-fit / val-scored only.

**`artifacts/`** – fitted models (each candidate frozen as tuned on val, plus `selected_model.joblib` / `.json` for `inference.py`). Git-ignored: not a required deliverable and rebuilt by running `detect_anomalies.py` (~40 s).

**`outputs/` – val evidence for the comparison (what the design doc and README cite)**

| File | Contents | Why it exists |
|---|---|---|
| `alerts_<model>_val.csv` (×4) | One row per alert: sensor, model, start, end, reason | The task's "detected anomalies" output; lets a reviewer check each alert and its explanation against `anomaly_labels.csv` |
| `val_comparison.csv` | One row per model: detected, misses, FA (total, per day, cross-sensor, benign), total errors, median delay, size, ms/minute, tuned params, explainability | The selection table (one model by total errors) |
| `val_per_sensor.csv` | Same metrics per model × sensor | Per-sensor reporting required by the selection rule |
| `val_per_type.csv` | Detected and median delay per model × anomaly type | Per-type recall and delay (reviewer fix: per-sensor medians hid per-type differences) |
| `val_tuning_log.csv` | Every grid setting tried, per model × sensor, with hits / misses / FA / errors / delay | Transparency of tuning: shows how sensitive each choice is (e.g. IF door p99.0 = 24 FA) |
| `val_chart_reactions.csv` | Best EWMA vs best CUSUM setting per sensor | Answers "how does each sensor react to each chart" |
| `val_sensitivity.csv` | Rules per sensor with the main vs the wide schedule | Quantifies the circularity caveat (reviewer fix); not used for selection |

`outputs/eda/` (Stage 4 plots) is written by `analysis.ipynb`, not by this script.

### Stage 5 implementation notes and corrections
- **Code:** all four models live in `detect_anomalies.py` (sections 2a rules, 2b IF v1/v2, 2c charts), share the alert/scoring/tuning code, and save `artifacts/candidate_<model>.joblib`; the selected one is copied to `selected_model.joblib` (+ `.json`). Tie order for selection: rules → charts → IF v2 → IF v1 (simpler first).
- **Bug fixed before reporting:** CUSUM did not restart after a signal; after the 01‑26 heating failure the lower sum stayed above the limit for 2.5 days, giving one alert that covered both failures and hid false alarms. Fixed (restart at 0 after a signal).
- **Reviewer fixes:** (1) circularity caveat written down + schedule sensitivity run; (2) delay now reported per anomaly type. My first chart summary wrongly said heating-failure delay was "about the same as rules"; per-type delays show charts ~27 min vs rules ~59 min.
- **Chart signal correction (design, not val tuning):** first version also charted `temp_resid`; 9 of its 10 temp FAs were heat-up ramps (06:30–07:10, 17:25–18:15), where readings sit above their lagging 15-min median for many minutes and the chart accumulates them. A chart measures sustained shifts; a 1–3 min spike is a single-minute rule's job, so `temp_resid` was removed and temp charts `temp_dev` (lower side) only. Effect: temp 4/4 hits 10 FA → 2/4 hits (spikes not targeted) 1 FA; charts overall 14 → 7 errors.
- **Not iterated further (by decision):** IF v2 is the single fair revision; chart constants (λ, k, floors) and grids were not changed after seeing val.

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
- **Rule thresholds from unlabelled data:**
  - Fridge power loss at < 1 W sits in an empty valley (0 train minutes in 0.75–1.0 W, between the ~0 W cluster and idle at ≥ 1.0 W).
  - Stuck-on at > 30 min sits in a gap (q99 of running periods = 22 min, none between 22 and 60 min).
  - Show briefly as evidence that no labels were used.
- **Explainability extension:** IF alerts currently report the most extreme feature vs train. Future work: SHAP values (e.g. `shap.TreeExplainer` on each IF) for per-feature contributions.
- **Schedule vs learned context:** the rule baseline uses a user-configured away/night schedule; the ML models get no schedule and learn context from the data.
- **Circularity caveat (must be stated):** the rule schedule matches the injection windows, so rule recall is an upper bound. Show the schedule sensitivity result (wide schedule: 14/14, 2 FA vs 0) and that a user-entered schedule is a deployment risk.
- **Model comparison (Stage 5):** rules, IF v1, IF v2 (fewer sensor-specific inputs + learned activity rates), EWMA/CUSUM control charts. Show the val comparison table and the per-type delay table. Key points: charts are fastest on gradual/sustained anomalies (heating failure ~21 vs ~59 min, power loss, stuck-on) and miss short night door openings; they target shifts only (spikes left to rules, after `temp_resid` on heat-up ramps caused 9 FA); IF improves with focused inputs but still misses duration/context anomalies; only rules catch the context-dependent ones, helped by the schedule.
- **Process notes for Stage 5:** CUSUM restart bug found and fixed before reporting; reviewer caught a misreported delay (per-sensor median hid per-type differences), so delays are now reported per type.
- **Possible extension:** a hybrid (charts for fridge/heating drift, rules for context-dependent door/motion) — not built.
- **Heating failure is contextual:** normal setback cooling (~0.6 °C/h at its start) overlaps the slowest failures. The rate of fall alone is not enough; the detector needs the context (target 21 °C, awake and home).
- **Productionization:** on-device serving. Cloud is used for retraining and model updates. Monitoring and retraining strategy still to be written.
- **AI transcripts:** exported per phase into `ai_transcripts/`.
- **Repo layout:** the tree and a one-line role for each file (Stage 2).
- **Setup & run:** `pip install -r requirements.txt`, then the run order `generate_data.py` → `analysis.ipynb` → `detect_anomalies.py` → `inference.py`. Note that the committed data is reproducible from the seed in `config.py`.
