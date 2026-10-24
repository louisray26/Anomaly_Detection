"""Score the frozen val-tuned models on the test split, once (docs/design.md, Stage 6).

Nothing is fit or tuned here: every model, threshold, baseline and rate is loaded from
artifacts/ as saved by detect_anomalies.py. Test labels are used for scoring only.

Pipeline:
    1. Test features (warm-up from val; baseline and activity rates from train via artifacts)
    2. Minute flags -> alerts per model (same code as val)
    3. Scoring (same rules as val) + schedule sensitivity for the rules
    4. Save test tables and alerts to outputs/, print test next to val

Run:  python evaluate_test.py   (after detect_anomalies.py)
"""

import joblib
import numpy as np
import pandas as pd

import config as cfg
import detect_anomalies as D
import features as F

MODELS = ["rules", "charts", "iforest_v2", "iforest"]
N_TUNED = {"rules": 3, "charts": 8, "iforest_v2": 4, "iforest": 4}


# ============================================================
# 1. Data
# ============================================================
def load_artifacts():
    return {m: joblib.load(cfg.ARTIFACTS_DIR / f"candidate_{m}.joblib") for m in MODELS}


def load_test(artifact):
    """Test features with the train-fit baseline and activity rates stored in the artifact."""
    test = F.split_features("test", artifact["baseline"])
    return D.add_activity_rates(test, artifact["activity_rates"])


# ============================================================
# 2. Alerts per model (frozen settings)
# ============================================================
def model_flags(name, art, test, sensor, schedule=None):
    """Minute flags for one sensor from a frozen artifact."""
    if name == "rules":
        return D.rule_flags(test, sensor, art["params"][sensor], schedule or art["schedule"])
    if name == "charts":
        return D.chart_flags(test, sensor, art["stats"], art["settings"][sensor])
    return D.iforest_flags(test, art["models"][sensor], sensor, art["thresholds"][sensor])


def model_alerts(name, art, test, sensor, schedule=None):
    flags = model_flags(name, art, test, sensor, schedule)
    explain_fn = None
    if name.startswith("iforest"):
        explain_fn = lambda m: D.explain(test, art["models"][sensor], sensor, m)
    return D.flags_to_alerts(flags, sensor, name, explain_fn)


# ============================================================
# 3. Scoring
# ============================================================
def score_model(name, art, test, labels, benign, schedule=None):
    """{sensor: (result, alerts)} in the same shape as the val tuning results."""
    results = {}
    for sensor in D.SENSORS:
        alerts = model_alerts(name, art, test, sensor, schedule)
        results[sensor] = (D.score_sensor(alerts, labels, benign, sensor), alerts)
    return results


# ============================================================
# Main
# ============================================================
def main():
    arts = load_artifacts()
    test = load_test(arts["charts"])   # baseline + activity rates (fit on train)
    labels = F.load_labels()
    labels = labels[labels["split"] == "test"]
    log = F.load_generation_log()
    benign = log[(log["split"] == "test") & log["kind"].isin(D.BENIGN_KINDS)]
    n_days = len(test) / (24 * 60)

    results, rows = {}, []
    for name in MODELS:
        art = arts[name]
        results[name] = score_model(name, art, test, labels, benign)
        size = (cfg.ARTIFACTS_DIR / f"candidate_{name}.joblib").stat().st_size / 1024
        ms = D.time_per_minute(lambda: [model_flags(name, art, test, s) for s in D.SENSORS], len(test))
        rows.append(D.summarise(name, results[name], n_days, size, ms, N_TUNED[name], ""))
    comparison = pd.DataFrame(rows).drop(columns="explainability")
    per_sensor = pd.concat([D.per_sensor_table(n, r) for n, r in results.items()])
    per_type = pd.concat([D.per_type_table(n, r) for n, r in results.items()])

    # Sensitivity: rules with the wide schedule, val-tuned params unchanged (reported only)
    wide = score_model("rules", arts["rules"], test, labels, benign, cfg.SCHEDULE_SENSITIVITY)
    sensitivity = pd.concat([D.per_sensor_table("rules (main schedule)", results["rules"]),
                             D.per_sensor_table("rules (wide schedule)", wide)])

    # Outputs
    for name, res in results.items():
        alerts = [a for _, (_, al) in res.items() for a in al]
        pd.DataFrame(alerts).sort_values("start").to_csv(cfg.OUTPUTS_DIR / f"alerts_{name}_test.csv", index=False)
    comparison.to_csv(cfg.OUTPUTS_DIR / "test_comparison.csv", index=False)
    per_sensor.to_csv(cfg.OUTPUTS_DIR / "test_per_sensor.csv", index=False)
    per_type.to_csv(cfg.OUTPUTS_DIR / "test_per_type.csv", index=False)
    sensitivity.to_csv(cfg.OUTPUTS_DIR / "test_sensitivity.csv", index=False)

    # Report: test next to val
    pd.set_option("display.width", 220)
    val = pd.read_csv(cfg.OUTPUTS_DIR / "val_comparison.csv")
    cols = ["model", "detected", "false_alarms", "benign_fa", "cross_sensor_fa", "total_errors"]
    both = val[cols].merge(comparison[cols], on="model", suffixes=("_val", "_test"))
    print("Val vs test:\n", both.to_string(index=False))
    print("\nTest comparison:\n", comparison.to_string(index=False))
    print("\nTest per sensor:\n", per_sensor.to_string(index=False))
    print("\nTest per anomaly type (detected | median delay min):\n",
          per_type.pivot(index=["sensor", "anomaly_type"], columns="model",
                         values=["detected", "median_delay_min"]).to_string())
    print("\nTest sensitivity (rules, wide schedule):\n", sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
