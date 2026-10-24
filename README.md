# Smart-Home IoT Anomaly Detection (PoC)

A proof of concept that detects anomalies in four smart-home sensors (living-room temperature, fridge power, hallway motion, front door). It uses 28 days of synthetic 1-minute data. Detection is **causal and per sensor**, so it can run in real time on an assumed home hub. Four detectors are compared: rules, two Isolation Forests and EWMA/CUSUM control charts. Labels are used only for evaluation.

Full decision log: [`docs/design.md`](docs/design.md). This README is the short version.

---

## 1. Overview

- **Data:** a simulated two-adult household in winter, with 4 sensors at 1-minute sampling over 28 days. It has 7 injected anomaly types, benign rare events (distractors) and realistic data-quality problems.
- **Split by time:** train (days 1–14), validation (15–21), test (22–28). Thresholds are tuned on validation. Test was scored **once**, with frozen models (git tag `frozen-before-test`).
- **Pipeline:**
  - Cleaning and features are causal and shared (`features.py`).
  - Each sensor has its own detector.
  - Flagged minutes are merged into alerts.
  - Each alert carries a readable reason.
- **Models compared:** rule baseline, Isolation Forest v1, Isolation Forest v2 (focused inputs), and EWMA/CUSUM control charts.
- **Result on test:**

| Model | Detected | False alarms | Total errors | Size | ms / minute |
|---|---|---|---|---|---|
| Rules (selected on val) | 14/14 | 0 | **0** | 3 KB | 0.0005 |
| Control charts | 12/14 | 3 | 5 | 14 KB | 0.006 |
| Isolation Forest v2 | 8/14 | 4 | 10 | 20.9 MB | 0.027 |
| Isolation Forest v1 | 2/14 | 4 | 16 | 11.5 MB | 0.023 |

The rules' perfect score is an **upper bound**. The rules use a configured schedule that matches how the data was generated (see [§6](#6-models-and-justification)). The production recommendation is a **hybrid** chosen per sensor (§9): EWMA control charts for fridge and heating drift, rules for temperature spikes, door and motion. It is assembled from the evaluated per-sensor detectors and was not scored as one combined system.

---

## 2. Setup and run

Tested with Python 3.15 on Windows.

```bash
python -m venv .venv
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Run in this order:

```bash
python generate_data.py         # data/: train/val/test CSVs, anomaly_labels.csv, generation_log.csv
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb   # cleaning checks + EDA (optional)
python detect_anomalies.py      # fit on train, tune + compare on val, save models to artifacts/
python evaluate_test.py         # score the frozen models on test, once
```

- The data in `data/` is committed. `generate_data.py` regenerates it from the seed in `config.py` (`SEED = 42`).
- `artifacts/` (fitted models) is git-ignored. `detect_anomalies.py` rebuilds it, and `evaluate_test.py` needs it.
- The notebook can also be opened in VS Code or Jupyter. The `notebook` package is not pinned, because it failed to install on Python 3.15.

### Repo layout

```
├── config.py             # seed, dates, splits, sensors, generator + detector parameters
├── generate_data.py      # synthetic data, labels and generation log -> data/
├── features.py           # shared causal cleaning + features (single source of truth)
├── analysis.ipynb        # data-quality checks, EDA plots, feature checks
├── detect_anomalies.py   # 4 detectors: fit on train, tune on val, compare, select
├── evaluate_test.py      # one-time test scoring of the frozen models
├── data/                 # readings (train/val/test.csv), anomaly_labels.csv, generation_log.csv
├── outputs/              # alerts_<model>_<split>.csv, val_*.csv, test_*.csv, eda/
├── docs/                 # task.md, plan.md, design.md (full decision log)
└── ai_transcripts/       # unedited AI session transcripts, one per phase
```

**Detected anomalies output:** `outputs/alerts_<model>_<split>.csv`, one row per alert with the columns `sensor, model, start, end, reason`.

---

## 3. Synthetic data

**Household (drives every sensor):** two adults. Adult A works in the office Mon–Fri. Adult B is hybrid: office Tue–Thu, home on Mon and Fri. The house is empty only on Tue–Thu daytime. At weekends one adult is always home. Wake-up is around 06:30 and sleep around 23:00, with daily jitter.

| Sensor | Normal behaviour |
|---|---|
| `temp_living_room` | Thermostat set to 21 °C when awake and home, 18 °C at night and in eco mode (empty house). The room follows the setting with a lag (45 min heating, 300 min cooling). Afternoon sun adds about 0.5 °C; sensor noise is 0.1 °C |
| `power_fridge` | Compressor 100–130 W for 12–18 min, idle 2–5 W for 18–30 min. Defrost at about 300 W every ~10 h. Runs are longer at mealtimes |
| `motion_hallway` | Probability per minute follows occupancy: 0.10–0.30 when home and awake, 0.005 asleep, 0.001 when away (stray triggers). Motion in the first and last minute of each door opening |
| `door_front` | Opens for 1–3 min at each departure or arrival, plus 0–2 extra openings per day |

**Events vs anomalies:** the two terms are kept strictly separate.
- An **event** is normal activity and is never labelled. Routine events are departures, compressor cycles and defrosts. Benign rare events are night bathroom trips (~3 per week) and one weekend-guests evening per split. The benign rare events are there to test false alarms.
- An **anomaly** is injected, labelled and is the detection target.

**Data-quality issues, which are not anomalies:**
- in each split, one outage per sensor (10–120 min) and one hub outage (20–60 min, rows missing);
- duplicate rows;
- 0.1% dropped readings.

These are handled in cleaning and excluded from scoring.

**Anomalous minutes:** 7.3% (train), 9.8% (val), 9.2% (test). This is higher than a real home, mainly because heating-failure labels include the recovery. Train contains unlabelled anomalies, which is realistic contamination for unsupervised fitting.

---

## 4. Anomaly definitions

An anomaly is an observation that is unlikely given its **context**: time of day, day of week, duration, or consistency with another sensor. The same reading can be normal or anomalous; for example, a door opening at 08:15 is normal and at 03:10 it is not.

Each type below has 2 anomalies in val and 2 in test, and at least 3 in train (unlabelled for the models).

| Type | Sensor | Real-world meaning | Injection |
|---|---|---|---|
| `temp_spike` | temp | Sensor glitch | 1–3 min jump of ±3–8 °C |
| `heating_failure` | temp | Boiler fault | Falls 0.5–1.0 °C/h for 1.5–3 h (≥ 1.5 °C total) while the setting is 21 °C. Label includes the recovery. **Contextual:** normal night/eco cooling starts at ~0.6 °C/h, overlapping the slowest failures, so the rate alone is not enough; it matters only when the room should be at 21 °C |
| `fridge_power_loss` | fridge | Unplugged / failed | 0–0.5 W for 45–120 min |
| `compressor_stuck_on` | fridge | Door ajar / refrigerant leak | Constant 100–130 W for 1–2 h (a normal run is ≤ 18 min) |
| `intrusion_motion` | motion | Intruder | Motion in 30–60% of minutes for 5–30 min while the house is empty (Tue–Thu), with no door opening |
| `door_open_night` | door | Security breach | Door open 1–5 min between 00:30 and 05:00 |
| `door_left_open` | door | Forgot to close | Door open 20–60 min while someone is home |

---

## 5. Cleaning and features

All steps are **causal**: each value uses only past and current data, so the same code serves batch evaluation and a live stream.

- **Cleaning:**
  - Sort, drop duplicates, and reindex to a 1-minute grid.
  - Physically impossible values become NaN (fixed hardware limits).
  - Gaps of up to 2 min are forward-filled.
  - Longer gaps mark the sensor **offline**, and its detector skips those minutes.
  - Outliers are **not** clipped, because they are the targets.
- **Features:**

| Sensor | Features |
|---|---|
| Temperature | `temp_dev`: reading minus the usual value for this hour (learned on train, two day groups). `temp_resid`: reading minus the 15-min rolling median. Rates over 15 and 60 min, computed on the rolling median |
| Fridge | Mode from fixed physical thresholds (idle < 50 W, running 50–200 W, defrost ≥ 200 W) and minutes in the current mode |
| Motion | Motion minutes in the last 15 / 60 min. `minutes_since_door_open` is the cross-sensor link for intrusion; it is NaN if the door sensor was offline since the last opening |
| Door | Minutes the door has been open |

- **Leakage rules:**
  - Everything learned (the temperature baseline, activity rates, chart centres and scales) is fit on **train only**.
  - Labels are never model inputs.
  - Label-split statistics in the EDA are descriptive only; thresholds never come from them.
- **Split warm-up:** val and test features are built with 180 min of history from the previous split. The first hour is therefore identical to a continuous run (verified in `analysis.ipynb`).

---

## 6. Models and justification

All models use **one detector per sensor**. This keeps alerts attributed and explainable, and an offline sensor only disables its own detector. Flagged minutes less than 10 min apart are merged into one alert. The alert time is its first flagged minute.

| Model | How it works | What is learned / tuned |
|---|---|---|
| **Rules** | Spike: \|`temp_resid`\| > 1.5 °C. Heating failure: below usual and falling, held for 15 min. Fridge: < 1 W for 10 min, or running > 30 min. Intrusion: motion in the configured *away* hours with no door opening for D min. Door: open in the configured *night* hours, or open > 10 min | Thresholds from physics, house rules or gaps in **unlabelled** train data (e.g. no train fridge minutes between 0.75 and 1.0 W; no train compressor runs between 22 and 60 min). 3 values tuned on val |
| **Isolation Forest v1** | One forest per sensor on its features plus hour and day-type columns | Threshold = percentile of train scores, tuned on val |
| **Isolation Forest v2** | 2–3 sensor-specific inputs. Context comes as a learned activity rate per (day type, hour) instead of generic time columns. `max_samples` = 1024 | Same as v1 |
| **Control charts** | EWMA or CUSUM on standardised signals: temperature below usual (`temp_dev`), fridge power and run length per mode, motion/door relative to their learned hourly rate. Charts measure sustained **shifts**; spikes are left to the rules | Centres, scales and rates from train. Chart type and limit per sensor tuned on val (λ = 0.2, k = 0.5 fixed) |

**Trade-offs: why these four were compared**

| | Rules | Control charts | Isolation Forest v2 | Isolation Forest v1 |
|---|---|---|---|---|
| **Role** | Simple, explainable baseline | Classic lightweight drift detector | Fair revision of v1 (focused inputs) | Standard unsupervised method for tabular data |
| **Errors (val / test)** | 0 / 0 (upper bound) | 7 / 5 | 9 / 10 | 13 / 16 |
| **Strength** | Catches every type, including context-dependent ones (intrusion, night door) | Fastest on gradual and sustained anomalies (heating ~21 vs ~59 min, fridge) | Better than v1 with the same false alarms; catches spikes and door left open | Generic: one method for every sensor, no hand-written rules; learns from unlabelled data |
| **Weakness** | Designer bias: thresholds and schedule mirror the simulator | Misses short or purely time-of-day anomalies; false alarms when temperature runs below its usual level | Misses duration and context anomalies (stuck-on, intrusion, night door) | Inputs diluted by context columns; scores each minute independently |
| **Needs a schedule** | Yes, hand-written away/night windows | No (learned rates) | No (learned rates) | No (hour / day-type columns) |
| **Size / speed** | 3 KB · 0.0005 ms/min | 14 KB · 0.006 ms/min | 20.9 MB · 0.027 ms/min | 11.5 MB · 0.023 ms/min |
| **Explainability** | Rule + threshold ("compressor running > 30 min") | Chart + signal + side ("EWMA of temp_dev below −6 σ") | Most extreme feature vs train (SHAP as extension) | Same as v2 |
| **Maintenance** | Manual: user/engineer updates the schedule and thresholds when routines change | Automatic: refit KB-sized centres, scales and rates; limits stay | Refit MB-sized forests and re-check the threshold | Same as v2 |

**Why these four:**
- **Rules** are the simple, explainable baseline: tiny and instant, and each alert reads like "compressor running > 30 min". **Weakness:** they need a hand-written away/night schedule, and the same person designed both the rules and the simulator.
- **Isolation Forest** is the standard unsupervised method for tabular data. **Weaknesses:** it scores each minute independently, it is diluted by context columns (v1), and it is the largest and slowest model.
- **Control charts** are a classic, lightweight way to detect gradual drift, with no hand-written schedule. **Weakness:** short or purely time-of-day anomalies give them little evidence to accumulate.

**Models not tried: forecasting and sequence models** (e.g. ARIMA/Prophet residuals, LSTM or autoencoders).
- **Little data:** 14 days of train, with unlabelled anomalies in it, is too little to fit a sequence model reliably.
- **Sensor types:** forecasting suits continuous series, and only temperature is one. Motion and door are binary, and fridge power is a cyclic on/off signal that is better described by its mode and run length.
- **Hardware:** the target is an assumed home hub, and Isolation Forest was already the largest model (11–21 MB). A sequence model would be larger and slower.
- **Explainability:** alerts must make sense to a non-expert user. Rules and charts give a direct reason; a forecast error from a neural network does not.

**Tuning and selection:**
- **Objective:** misses and false alarms are equally costly. Each sensor is tuned to minimise **misses + false alarms** on val. Ties go to the lower detection delay, then to the more conservative setting.
- **Selection:** one model is selected by total val errors. **Rules were selected:** 0 val errors, against 7 for charts, 9 for IF v2 and 13 for IF v1.

**Caveat:** the rules' score depends on a configured schedule (away Tue–Thu 09–17, night 00–05). That schedule matches the windows the generator used to inject intrusions and night openings, so **rule recall is an upper bound**. The ML models and charts get no schedule. A sensitivity run widened only the schedule (away 08:00–18:30, night 23:30–06:00). Errors rose from 0 to 2 on val and from 0 to 1 on test, because people are still home at 08:00 before they leave.

**Production recommendation: a hybrid** (details in §9). It is assembled from the evaluated per-sensor detectors and was not scored as one combined system.
- **EWMA control charts for fridge power and heating drift.** They were faster on val (heating failure ~21 vs ~59 min, power loss 4.5 vs 9 min) and need no hand-written schedule.
- **Rules for temperature spikes, door and motion.** Spikes are not a chart target, and door and motion anomalies depend on the time of day, which charts cannot see.
- **A further extension:** chart residuals from a context-adjusted expected value instead of raw readings.

---

## 7. Evaluation protocol

- **Hit:** an alert on the anomaly's own sensor that overlaps [start, end + 15 min grace].
- **False alarm:** any other alert on that sensor. Two kinds of false alarm are also reported separately:
  - *cross-sensor*: during another sensor's anomaly;
  - *benign*: on a logged benign event.
- **Delay:** first alert minus anomaly start, reported **per anomaly type**. Per-sensor medians mix spikes (0 min) with slow failures.
- **Test once:** `evaluate_test.py` loads the frozen val-tuned artifacts and fits nothing. The model code is unchanged since tag `frozen-before-test` (commit `3b32d9e`). You can check this with `git diff frozen-before-test -- detect_anomalies.py features.py config.py`.
- **No changes after test:** test results are reported as they are.

---

## 8. Results

**Val vs test (total errors = misses + false alarms):**

| Model | Val detected | Val FA | Val errors | Test detected | Test FA | Test errors |
|---|---|---|---|---|---|---|
| Rules | 14/14 | 0 | 0 | 14/14 | 0 | **0** |
| Control charts | 9/14 | 2 | 7 | 12/14 | 3 | **5** |
| Isolation Forest v2 | 8/14 | 3 | 9 | 8/14 | 4 | **10** |
| Isolation Forest v1 | 4/14 | 3 | 13 | 2/14 | 4 | **16** |

The val ranking holds on test.

**Test: detected · median delay (min) per anomaly type:**

| Anomaly type | Rules | Charts | IF v2 | IF v1 |
|---|---|---|---|---|
| temp_spike | 2/2 · 0 | 1/2 · 0 | 2/2 · 0 | 1/2 · 0 |
| heating_failure | 2/2 · 54.5 | 2/2 · **19** | 1/2 · 115 | 1/2 · 134 |
| fridge_power_loss | 2/2 · 9 | 2/2 · **3.5** | 0/2 | 0/2 |
| compressor_stuck_on | 2/2 · 25.5 | 2/2 · **18.5** | 2/2 · 71.5 | 0/2 |
| intrusion_motion | 2/2 · **2.5** | 1/2 · 17 | 1/2 · 20 | 0/2 |
| door_open_night | 2/2 · **0** | 2/2 · 3 | 0/2 | 0/2 |
| door_left_open | 2/2 · 10 | 2/2 · 5.5 | 2/2 · **5** | 0/2 |

**Reading the results:**
- **Control charts** are the fastest detector on gradual and sustained anomalies (fridge, heating failure, door left open). Their 3 test false alarms are temperature below its usual level.
- **Rules** are the only reliable detector for context-dependent anomalies (intrusion, night openings), helped by the schedule.
- **Isolation Forest** improved with focused inputs (v1 → v2), but still misses duration and context anomalies.
- **Small samples:** with 2 anomalies per type, a difference of one is within noise. The overall ranking is the robust result.

Per-sensor tables, tuning logs and every alert with its reason are in `outputs/`.

---

## 9. Productionization

**Production recommendation: the hybrid.**
The PoC selected the rules on val. Their score is an upper bound, because their schedule matches the generator, so the production recommendation is the hybrid from §6, chosen per sensor:

| Sensor | Detector | Why |
|---|---|---|
| Fridge power | EWMA control chart (power and run length per mode) | Faster on val (power loss 4.5 vs 9 min, stuck-on 19 vs 26 min); no schedule needed |
| Temperature | EWMA chart on `temp_dev` for heating drift + the spike rule (\|`temp_resid`\| > 1.5 °C) | Faster on heating failure (~21 vs ~59 min on val), at the cost of more false alarms (val 1 vs 0, test 3 vs 0 for rules on temperature); charts do not target spikes |
| Motion | Rules (intrusion: away hours + no door opening) | The anomaly depends on time of day and the door link; charts cannot see it |
| Door | Rules (night opening, left open > 10 min) | Same: night openings are short and purely time-of-day |

- **Built from evaluated parts:** each detector is the val-tuned component already saved. Detectors are independent per sensor, so the hybrid is a per-sensor choice and needs no new model.
- **Not evaluated as a combined system:** the temperature combination (chart plus spike rule) was never scored together. The hybrid was chosen from val results, not from test, and should be validated on fresh data before release (the release gate below).
- **Next step:** chart residuals from a context-adjusted expected value instead of raw readings.

**Serving: real time, on-device.**
- The detectors run on the assumed home hub (Raspberry Pi / Home Assistant class). Readings arrive once a minute, and each detector updates its state and alerts immediately.
- **Why on-device:** this is a security and appliance use case. Alerts must work without internet, and raw in-home activity data stays private.
- **Same code as evaluation:** `features.py` is causal and shared, so the stream is designed to run the same code that was evaluated. This needs only a rolling buffer of the last 180 minutes, the frozen artifact (3–14 KB for rules or charts), and per-detector chart state.
- **Not verified:** a streaming-equals-batch check was not run in this PoC (Stage 7 was skipped by decision). It would be the first test to add.
- **Batch use is secondary:** for example, a nightly summary of alerts and data quality.

**Monitoring (on the hub, with summaries to the cloud):**
- **Alert rate** per sensor per day, against the val baseline. A sudden rise or a silent sensor is itself a signal.
- **User feedback** on each alert (confirm / dismiss). This gives a running precision estimate and, over time, real labels.
- **Input drift:**
  - temperature deviation from the learned baseline;
  - fridge duty cycle and run lengths;
  - hourly motion and door activity rates compared with the train-time rates.
- **Data quality:** offline minutes, gaps and duplicates per sensor. A dying sensor should raise a maintenance message, not anomaly alerts.
- **Misses are invisible without labels.** Cover this by asking the user after unconfirmed incidents, and by periodically replaying synthetic injected anomalies through the deployed detectors.

**Retraining (refit on the hub, release managed in the cloud):**
- **Refit on the hub:** only small statistics are refit (temperature baseline, activity rates, chart centres and scales). They are fit on a rolling window of recent weeks, excluding confirmed-anomaly windows. The refit is cheap enough for the hub, and raw in-home data stays on the device.
- **Triggers:**
  - a schedule (e.g. monthly);
  - drift alarms;
  - known changes: heating season vs summer, daylight-saving shifts, a new routine or household member.
- **Release gate in the cloud:** the hub uploads the candidate statistics (KB-sized, not raw data). The cloud replays injected anomalies and the user-confirmed incidents through the candidate. A candidate that passes then runs in **shadow mode** on the hub next to the live model, and ships only if errors do not increase.
- **Versioned artifacts and rollback in the cloud:** every released artifact is versioned with its gate results, so the hub can be rolled back to the last good version.
- **Schedule-based rules need care:** prompt the user to confirm their away/night schedule when learned activity no longer matches it.

---

## 10. Limitations and future work

- **Synthetic data, one designer:** the simulator and the rules share assumptions, so all scores are optimistic. Real homes are noisier: pets, irregular schedules, multiple rooms.
- **Benign events never reach the rule triggers:**
  - Night bathroom trips are motion without a door opening, outside the away hours.
  - Weekend guests' door openings are short and happen 19:00–23:30, outside the night window.
  - So the rules' 0 false alarms on benign events is also partly by construction. Real distractors (a late-night delivery, a cleaner on an office day) would hit the triggers.
- **Data-quality issues never overlap anomalies:** outages were placed apart from anomalies to keep labels unambiguous. Real failures often come with outages: a power cut stops the fridge *and* the hub, and a dying sensor drifts before it goes offline. The detectors skip offline minutes, so an anomaly during an outage would be missed or detected late.
- **Small evaluation:** 14 anomalies per split, so per-type recall can only be 0, 50 or 100%.
- **One season, one household, no daylight-saving change.** The baselines would need refitting across seasons.
- **Contaminated train:** unlabelled anomalies in train make long fridge runs and door openings look less rare to Isolation Forest.
- **Future work:**
  - scoring the hybrid as one combined system, and context-adjusted chart residuals (§6, §9);
  - a Bernoulli or Poisson CUSUM on motion/door counts. The current charts standardise the binary readings as if they were Gaussian, so the chart limits have no exact false-alarm meaning. A count-based likelihood-ratio CUSUM fits binary and count data properly;
  - SHAP explanations for Isolation Forest (`shap.TreeExplainer`) instead of the "most extreme feature" explanation;
  - a streaming-equals-batch test;
  - multi-sensor rooms and a learned household schedule to replace the configured one.

---

## 11. Process and AI usage

The work followed a stage-by-stage plan ([`docs/plan.md`](docs/plan.md)). At each stage the AI assistant (Claude Code) proposed options and trade-offs, I decided, and the decisions were logged in [`docs/design.md`](docs/design.md) before moving on. The stages were:
1. problem understanding;
2. repo structure;
3. dataset design, creation and split;
4. cleaning and EDA;
5. model comparison;
6. evaluation and selection, ending with the single test run (`evaluate_test.py`);
7. production test: **skipped** by decision, because the task asks only for a production discussion (§9);
8. README (this document, last).

**Independent review:** a second Claude Code session acted as an independent reviewer of the data, features and implementation at several points. I verified each of its findings against the data or the outputs before anything was fixed, and the fix was agreed before it was applied. Corrections are documented rather than hidden:
- **Simulator realism:** fridge idle noise, cooling lag, anomaly durations, motion on arrival, stray motion, heating-recovery labels, door/motion timing, intrusion motion count.
- **Features:** a start-up burn-in, a split warm-up, a median-smoothed rate, a spike residual, a door-link feature, a pooled baseline, and NaN for door-dependent features when the door sensor was offline.
- **Models:**
  - A CUSUM restart bug, found before reporting: one heating failure kept the chart in alarm for 2.5 days.
  - A misreported heating-failure delay. Per-sensor medians had hidden it, so delays are now reported per type.
  - The undocumented schedule bias, which led to the sensitivity run.
  - Charting the spike residual caused heat-up false alarms, so charts now track shifts only.
- **Not iterated:** Isolation Forest v2 is the single planned revision. Nothing was changed after the test run.

**Transcripts** (`ai_transcripts/`, unedited exports):

| File | Phase |
|---|---|
| `001_problem_understanding.txt` | Problem understanding |
| `002_synthetic_data_generation.txt` | Repo structure, dataset design and generation |
| `003_cleaning_and_eda.txt` | Cleaning, features and EDA |
| `004_modelling_eval.txt` | Model comparison and validation |
| `005_readme_design.txt` | Test evaluation and README |
| `006_reviewer_check.txt` | External Reviewer |
| `final_transcript.txt` | Whole Workflow Session |
