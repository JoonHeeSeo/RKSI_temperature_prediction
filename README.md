# 🛰️ Incheon Airport (RKSI) Daily Temperature Prediction

[![Streamlit Demo](https://img.shields.io/badge/Demo-Streamlit-blue)](https://rksi-temperature-prediction.streamlit.app/)

https://rksi-temperature-prediction.streamlit.app/

## 📖 Overview

Predict **next‑day mean temperature** at **Incheon International Airport** (ICAO: **RKSI**) from historical weather observations using a suite of deep‑learning and classical models.

### Key Features

- **End‑to‑end pipeline**: data acquisition → preprocessing → model training → evaluation → interactive demo
- Multiple architectures: **Linear, MLP, LSTM, GRU, TCN, Transformer**
- Simple CLI interface for experimenting with individual or all models
- **Streamlit dashboard** for live inference, visualisations, and comparison with climatology baseline

---

## 🌐 Live Demo

```text
https://rksi-temperature-prediction.streamlit.app/
```

> or run locally with `streamlit run service/app.py` (see below).

---

## 🛠️ Installation

```bash
# Python 3.13+ is recommended
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r service/requirements.txt
```

---

## ⬇️ Data

Weather observations are pulled on‑demand from **[Meteostat](https://dev.meteostat.net/)** and cached locally.

```bash
python data/download_weather.py \
  --start 2023-01-01 \
  --end   2023-12-31 \
  --out   data/rksi_weather_2023.csv
```

---

## 🚀 Training

| Command                                | Description                    |
| -------------------------------------- | ------------------------------ |
| `python -m training.train_all`         | Train every model sequentially |
| `python -m training.train_linear`      | Linear Regression              |
| `python -m training.train_lstm`        | Long Short‑Term Memory (LSTM)  |
| `python -m training.train_mlp`         | Multi‑Layer Perceptron (MLP)   |
| `python -m training.train_gru`         | Gated Recurrent Unit (GRU)     |
| `python -m training.train_tcn`         | Temporal Convolutional Network |
| `python -m training.train_transformer` | Transformer encoder‑decoder    |

---

## 🖥️ Streamlit App

The dashboard lets you

- Pick a date range and compare model forecasts with actual observations
- Visualise uncertainty bands & residuals
- Download predictions as CSV

Run locally:

```bash
streamlit run service/app.py
```

Deploy effortlessly to **Streamlit Community Cloud** (or any Docker‑ready host).
