# Objective
Build a proof of concept that generates synthetic smart-home sensor data, detects plausible anomalies, evaluates detection quality, and documents the engineering decisions. This is a take home assigment for a senior ML scientist.

The assignment estimates 2–10 hours. Prioritize a correct, interpretable, well-documented solution over unnecessary complexity.

## Project Scope and Proposed Defaults

These are the working defaults. Confirm important assumptions before coding and document any approved changes.

- **Domain:** DIY smart-home anomaly detection.
- **Dataset duration:** 28 days.
- **Sampling interval:** One minute.
- **Maximum regular timestamps:** 40,320 per sensor before missing observations or outages.
- **Sensors:** Indoor temperature, refrigerator power, hallway motion, and front-door state.


# AI Agent Working Principles

- Act as my ML research and coding assistant. Collaborate with me, challenge my assumptions, and help me make informed technical decisions. I    remain responsible for reviewing and approving decisions.

- Work through one stage at a time. Propose the design, explain important trade-offs, and ask 3–5 focused questions when clarification is needed. Wait for my approval before implementing that stage.

- Do not raise topics that belong to later stages (repo structure, models, evaluation) until we reach them. Keep each reply short.

- Keep explanations concise and practical. Support recommendations with technical reasoning, evidence, or examples. Do not agree with me automatically; point out incorrect assumptions and material risks.

- Point out issues that are critcal or go against the task instrutions. Also when you provide the issues; put a prority on them.

- Base conclusions on actual data, experiments, and tests. Never fabricate results, metrics, execution logs, or test outcomes.

- Prevent data leakage, keep ground-truth labels separate from model inputs, and preserve the train/validation/test separation.

- Prioritize reproducibility, interpretability, readable code, and the simplest solution that meets the requirements. Use configurable parameters and random seeds where appropriate.

- Document important decisions, assumptions, trade-offs, results, and limitations in docs/design.md.

- Do not proceed to the next stage until I have reviewed and approved the current stage. Avoid repeating settled decisions or adding unnecessary complexity.

# Anamoly Detection Workflow

- Problem Understanding:
    Understand the real-time IoT anamoly detection for a DIY smart home system. 
    Undertand the differnce between an event and anamoly in given scenario. 
    Under the different types of sensor with real time examples.
    Decide on the type of events and anomalies you eant to inject in the datasets.

- Repo structure create
    Files:
    generate_data.py #Generates the train, validation and test dataset
    analysis.ipynb #Cleaning decsions & All graphs that needs to be visualized for undersatnding comes here
    detect_anomalies.py #Evaluation of all models and savimg the model happen here
    inference.py #Inferne with the best model - Production
    requirements.txt 
    ai_transcripts/
    docs/

- Dataset Design, Creation and Split
    Have identifers for the sensors, anomalies and others is needed
    Create generate_data.py to generate one dataset and split it chronologically
    Train: 1-14
    Validation: 15-21 (thresholds and model selection only)
    Test: 22-28 (run once at the end, after choices are final)

    Simulate daily cycles, weekly patterns, noise, realistic sensor ranges, and intermittent anomalies. 
    Generate ground-truth labels alongside the measurements, but keep those labels out of the model's input features.

    Important: Vary anomaly timing, magnitude, and duration across events, so the model can't learn one exact pattern..

- Cleaning & EDA 
    Check missing values, duplicates, gaps, and timestamp consistency. 
    Plot sensor values over time, daily and weekly profiles, and distributions. 
    Show injected events by anomaly type and sensor. 
    Document the number of injected events and affected observations.
    See if any preprocessing is required for the differnt models or differnt sensors

- Model comparison 
    Choose 2-3 differnt models
    - Rule based detector(Baseline):physical range checks, stuck-value checks, and time-of-day rules, with thresholds set from training data and sensor limits
    - Isolation Forest on time-aware features.
    - (others based on above results)
    Comparison should be made based on the evaluation metrics, model size, complexity.

- Evaluation and selection 
    How many injected events were detected? 
    How many events were detected on the correct sensor? 
    How many false alarms occurred? 
    How quickly were events detected? 
    Can an engineer understand why an alert was generated?

- Production-test 
    Stream the test period (days 22-28) line by line through the selected model, loading its saved preprocessing and configuration. Run this once only, after the model and thresholds are final. Report metrics by sensor and anomaly type.

- Write readme and organize all info after the project analysis is completed.
