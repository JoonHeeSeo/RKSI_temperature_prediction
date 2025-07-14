import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Model Metrics Comparison", layout="wide")
st.title("Model Performance Comparison")

METRICS_PATH = "service/results.csv"

# 1. Load metrics
if not os.path.exists(METRICS_PATH):
    st.error(f"Metrics file not found: {METRICS_PATH}")
    st.stop()

try:
    df = pd.read_csv(METRICS_PATH, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(METRICS_PATH, encoding='cp949')

# 2. Display raw table
st.subheader("Raw Metrics")
st.dataframe(df.set_index("model"))

# 3. Bar chart for MAE and RMSE
metrics_long = df.melt(id_vars="model", var_name="metric", value_name="value")
fig, ax = plt.subplots(figsize=(8, 4))
for metric, grp in metrics_long.groupby("metric"):
    ax.bar(grp["model"], grp["value"], label=metric, alpha=0.7)
ax.set_xlabel("Model")
ax.set_ylabel("Error (°C)")
ax.set_title("MAE and RMSE by Model")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.subheader("MAE & RMSE Comparison")
st.pyplot(fig)

# 4. Side-by-side bar charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("MAE by Model")
    fig_mae, ax1 = plt.subplots()
    ax1.bar(df["model"], df["MAE(℃)"], color="tab:blue")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("MAE (°C)")
    ax1.set_xticklabels(df["model"], rotation=45)
    plt.tight_layout()
    st.pyplot(fig_mae)

with col2:
    st.subheader("RMSE by Model")
    fig_rmse, ax2 = plt.subplots()
    ax2.bar(df["model"], df["RMSE(℃)"], color="tab:orange")
    ax2.set_xlabel("Model")
    ax2.set_ylabel("RMSE (°C)")
    ax2.set_xticklabels(df["model"], rotation=45)
    plt.tight_layout()
    st.pyplot(fig_rmse)