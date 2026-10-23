# Take-Home Exercise: Senior ML Scientist

## Objective

The goal of this exercise is to assess your ability to approach a loosely defined machine learning problem, make reasonable assumptions, and build a proof-of-concept solution. We want to see how you structure your work, how you think about the problem, and how you leverage modern tools to be effective. We are particularly interested in your process. Therefore, a key part of this exercise is to use an AI coding assistant (like Gemini, ChatGPT, GitHub Copilot, etc.) and submit the complete, unedited transcript(s) of your interactions along with your code.

**Estimated Time:** 2–10 hours. We trust you to manage your time effectively. The goal is not to spend the maximum time, but to produce a thoughtful and well-documented solution.

---

## The Scenario: IoT Anomaly Detection for a DIY Smart Home

Imagine you are building a service for a smart home enthusiast. Their house is equipped with numerous Internet of Things (IoT) sensors that collect various types of data (e.g., temperature, humidity, power consumption, motion detection, window/door states). Your task is to develop a proof-of-concept system that can detect anomalous events within the home.

An "anomaly" could be a security breach (a window opening unexpectedly at night), a potential appliance failure (a freezer's power consumption suddenly dropping to zero), or an environmental issue (a rapid, unexplained rise in humidity). The problem is intentionally open-ended. You will need to define what constitutes an "anomaly" and choose an appropriate ML/statistical approach to detect it.

---

## The Task

Your goal is to build a Python-based solution that can process time-series data from multiple sensors and flag potential anomalies.

### 1. Dataset Selection & Simulation

No real-world dataset is provided. You are expected to generate a synthetic dataset that realistically simulates a few different sensors in a smart home over a period (e.g., a few weeks).

**Requirements for the dataset:**

- It should be a time-series dataset (a CSV file is fine).
- Include at least 3–4 different sensor types (e.g., temperature_living_room, power_consumption_fridge, motion_detected_hallway, door_state_front).
- Simulate normal daily/weekly patterns (e.g., temperature changes with time of day, power consumption cycles, motion detection aligned with typical activity).
- Inject a few different types of plausible anomalies into the data. Be creative!

_Hint:_ Using an AI assistant to help generate this synthetic data is a great way to start.

---

### 2. Anomaly Detection Model

Using your synthetic dataset, build a model to identify the anomalies you've created.

- You have complete freedom to choose the modeling approach. It could range from simple statistical methods (e.g., rolling z-scores) to more complex unsupervised models (e.g., Isolation Forest, Autoencoders, LSTMs).
- The choice of model and your justification for it are key evaluation points. A simple, well-justified model is often better than a complex one that is poorly explained.

---

### 3. Deliverables

Please submit a link to a Git repository (e.g., GitHub, GitLab) containing the following:

#### 1. README.md

A well-written document that serves as the entry point to your project. It should include:

- A brief overview of your approach.
- Your definition of the anomalies you chose to model.
- Justification for your choice of anomaly detection model(s) and a discussion of its tradeoffs.
- Instructions on how to set up the environment (requirements.txt or similar) and run your code.
- A short discussion on how you would productionize this system. Please consider:
  - How would the model be served (e.g., batch vs. real-time)?
  - How would you monitor its performance over time, and how would you handle retraining?

#### 2. Source Code

A clean, well-commented Python project.

- A script to generate the synthetic dataset (`generate_data.py`).
- A script or Jupyter Notebook that trains/applies your model and outputs the detected anomalies (`detect_anomalies.py` or `analysis.ipynb`).

#### 3. AI Interaction Transcripts

A directory named `ai_transcripts/` containing the full, unedited chat logs from your interactions with AI coding assistants. Please export them as raw text, Markdown, or PDF.

---


