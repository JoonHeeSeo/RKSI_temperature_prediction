# RKSI Temperature Prediction

[![Streamlit Demo](https://img.shields.io/badge/Demo-Streamlit-blue)](https://rksi-temperature-prediction.streamlit.app/)

Predict next-day mean temperature at Incheon International Airport (ICAO: RKSI) using historical weather data.

**Live demo:** https://rksi-temperature-prediction.streamlit.app/

## Features

This project builds an end-to-end pipeline for temperature forecasting:

- Collect daily weather observations via Meteostat API
- Generate time-based features (month, seasonal cycles)
- Train and compare multiple time-series models
- Visualize results through a Streamlit dashboard

## Models

- Linear Regression
- MLP (Multi-Layer Perceptron)
- LSTM
- GRU
- TCN (Temporal Convolutional Network)
- Transformer

## Installation

```bash
uv sync
```

## Data

Download weather data from Meteostat:

```bash
uv run python data/download_weather.py \
  --start 2020-01-01 \
  --end   2023-12-31 \
  --out   data/rksi_weather.csv
```

## Training

Train all models:

```bash
uv run python -m training.train_all
```

Or train individually:

```bash
uv run python -m training.train_linear
uv run python -m training.train_mlp
uv run python -m training.train_lstm
uv run python -m training.train_gru
uv run python -m training.train_tcn
uv run python -m training.train_transformer
```

Results are saved to `service/results.csv`.

## Dashboard

Run the Streamlit app to compare model performance:

```bash
uv run streamlit run service/app.py
```

Features:
- Model performance comparison (MAE, RMSE)
- Predicted vs actual temperature plot

## Project Structure

```
├── data/                 # Data collection scripts
├── models/               # Model definitions
├── training/             # Training scripts
├── service/              # Streamlit app and results
├── utils/                # Feature engineering, metrics
└── checkpoints/          # Saved model weights
```
