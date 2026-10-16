# 💙 FlowSight AI — Health Operations Intelligence Platform

> **Datathon Track 1: Health — "The Forgotten Patient"**  
> Predicting and preventing patient delays using pure NumPy ML on real hospital operational data.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://flowsight-ai-nxehgkr7jqwnapmeaasvbt.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io)
[![NumPy](https://img.shields.io/badge/NumPy-ML%20from%20scratch-blue?logo=numpy)](https://numpy.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🌐 Live Demo

> **[https://flowsight-ai-nxehgkr7jqwnapmeaasvbt.streamlit.app/](https://flowsight-ai-nxehgkr7jqwnapmeaasvbt.streamlit.app/)**

Deployed on **Streamlit Community Cloud** — no installation needed, runs in your browser.

---


## 🏥 Overview

**FlowSight AI** is an executive-grade hospital operations intelligence dashboard that answers:

> *"What is likely to get delayed next, why, and what should operations examine?"*

Built on **15,012 genuine competition records** across 4 CSVs, with a **Pure NumPy Linear Regression** model trained via Batch Gradient Descent — zero external ML libraries.

---

## ✨ Features & Screens

| Screen | Description |
|---|---|
| 📊 **Operations Overview** | Real-time ED volume, triage mix, department snapshots, hourly heatmap |
| 🔮 **Delay Prediction Engine** | Interactive patient scenario simulator — predict wait time using the NumPy model |
| 🚨 **Bottleneck Investigator** | Rank worst operational bottlenecks by Impact Score on held-out test data |
| 👥 **Patient Demographics** | Age distribution, chronic conditions, insurance equity, sex × triage analysis |
| 🔬 **Model Lab & Diagnostics** | Calibration scatter, residual distribution, loss curve, feature weights |
| 🧪 **Diagnostic Test TAT** | TAT violin plots, monthly trends, SLA breach analysis by modality |

---

## 🤖 ML Model Architecture

```
Pure NumPy Multiple Linear Regression via Batch Gradient Descent

  y_hat = X @ w
  Loss  = (1/N) * Σ(y_hat - y)²          ← Mean Squared Error
  ∇     = (2/N) * Xᵀ(Xw − y)             ← Gradient
  w     ← w − η∇                          ← Weight update

  Learning Rate η = 0.05  |  Iterations = 1,500
```

**Held-Out Test Results (December 2025 — 697 visits):**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| NumPy Gradient Descent (Ours) | ~13.45 min | ~16.94 min | ~0.492 |
| Department Median Baseline | ~18.83 min | ~23.12 min | ~0.054 |
| Global Mean Naive | ~22.45 min | ~27.81 min | 0.000 |

> **+28.6% MAE improvement** over the department baseline, **zero data leakage**, temporal train/test split (Jan–Sep 2025 train → Dec 2025 test).

---

## 📁 Project Structure

```
flowsight_ai/
├── app.py                      # Main Streamlit application (6 screens)
├── requirements.txt            # Python dependencies
├── data/
│   ├── encounters.csv          # 8,000 patient encounters
│   ├── patients.csv            # 2,000 unique patients
│   ├── departments.csv         # 12 hospital departments
│   └── diagnostic_tests.csv   # 5,000 diagnostic test orders
├── src/
│   ├── data_loader.py          # Data loading & merging pipeline
│   ├── feature_engineering.py # FeaturePipeline (one-hot + scaling)
│   ├── model_from_scratch.py   # Pure NumPy LinearRegressionScratch
│   ├── baseline.py             # DepartmentMedianBaseline
│   ├── evaluation.py           # MAE, RMSE, R² metrics
│   ├── investigation_engine.py # Bottleneck Impact Score engine
│   └── visualizations.py      # Plotly chart helpers
├── assets/
│   └── *.jpg                   # Hospital imagery
└── tests/
    └── test_*.py               # 7 unit tests
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/jothsanthallu567-tech/flowsight-ai.git
cd flowsight-ai/flowsight_ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Or visit the **live deployed app** directly — no setup needed:

> 🚀 **[https://flowsight-ai-nxehgkr7jqwnapmeaasvbt.streamlit.app/](https://flowsight-ai-nxehgkr7jqwnapmeaasvbt.streamlit.app/)**


---

## 📊 Data Description

| File | Records | Key Fields |
|---|---|---|
| `encounters.csv` | 8,000 | arrival_time, wait_minutes, triage_level, outcome |
| `patients.csv` | 2,000 | age, sex, chronic_condition, insurance_type |
| `departments.csv` | 12 | department_name, staff_count, bed_count, service_type |
| `diagnostic_tests.csv` | 5,000 | test_type, priority, tat_minutes |

---

## 🛠️ Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
    <b>💙 FlowSight AI</b> — Operational decision-support only. NOT for clinical diagnosis.<br>
    Built with ❤️ for the Health Datathon.
</div>
