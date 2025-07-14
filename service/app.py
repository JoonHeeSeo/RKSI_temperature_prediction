from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from zoneinfo import ZoneInfo

# ──────────────────────────── Page & Header ─────────────────────────────

st.set_page_config(page_title="RKSI Temperature Prediction", page_icon="🌡️", layout="wide")

st.title("Model Performance Comparison")

today = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%B %d, %Y")
st.caption(f"Data as of: {today}")

# ──────────────────────────── Data Loading ──────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "results.csv"

@st.cache_data
def load_metrics(path: Path) -> pd.DataFrame:
    """Load CSV with utf-8 / cp949 fallback & cache it."""
    for enc in ("utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    st.error(f"Failed to read CSV: {path}")
    st.stop()

if not METRICS_PATH.exists():
    st.error(f"Metrics file not found: {METRICS_PATH}")
    st.stop()

df = load_metrics(METRICS_PATH)

# ──────────────────────────── Highlight Cards ───────────────────────────

best_mae  = df["MAE(℃)"].min()
best_rmse = df["RMSE(℃)"].min()

c1, c2 = st.columns(2)
c1.metric("🔹 Best MAE",  f"{best_mae:.3f} ℃")
c2.metric("🔸 Best RMSE", f"{best_rmse:.3f} ℃")

# ──────────────────────────── Raw Table ─────────────────────────────────

st.subheader("Raw Metrics")
st.dataframe(df.set_index("model"))

# ──────────────────────────── Scatter Plot ──────────────────────────────

st.subheader("MAE vs. RMSE (lower-left is better)")

fig_s, ax_s = plt.subplots(figsize=(6, 6))
ax_s.scatter(df["MAE(℃)"], df["RMSE(℃)"], s=120, alpha=0.8)

# Highlight best MAE point
best_idx = df["MAE(℃)"].idxmin()
ax_s.scatter(df.loc[best_idx, "MAE(℃)"],
             df.loc[best_idx, "RMSE(℃)"],
             s=240, color="crimson", edgecolor="black", zorder=5,
             label="Best MAE")

# Add labels with slight offset
for _, row in df.iterrows():
    ax_s.text(row["MAE(℃)"] + 0.005,
              row["RMSE(℃)"] + 0.005,
              row["model"],
              fontsize=9)

# Dynamic axis limits to reduce empty space
min_val = min(df["MAE(℃)"].min(), df["RMSE(℃)"].min())
max_val = max(df["MAE(℃)"].max(), df["RMSE(℃)"].max())
padding = (max_val - min_val) * 0.15  # 15% padding around data

ax_s.set_xlim(min_val - padding, max_val + padding)
ax_s.set_ylim(min_val - padding, max_val + padding)

# Reference diagonal
ax_s.plot([min_val - padding, max_val + padding],
          [min_val - padding, max_val + padding],
          ls="--", color="grey", alpha=0.5)

ax_s.set_xlabel("MAE (℃)")
ax_s.set_ylabel("RMSE (℃)")
ax_s.grid(True, ls="--", alpha=0.3)
ax_s.legend()
st.pyplot(fig_s)

# ──────────────────────────── Horizontal Bars ───────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("MAE by Model (sorted)")
    df_mae = df.sort_values("MAE(℃)")
    fig_mae, ax_mae = plt.subplots(figsize=(6, 4))
    ax_mae.barh(df_mae["model"], df_mae["MAE(℃)"], color="tab:blue")
    ax_mae.set_xlabel("MAE (℃)")
    ax_mae.invert_yaxis()
    st.pyplot(fig_mae)

with col2:
    st.subheader("RMSE by Model (sorted)")
    df_rmse = df.sort_values("RMSE(℃)")
    fig_rmse, ax_rmse = plt.subplots(figsize=(6, 4))
    ax_rmse.barh(df_rmse["model"], df_rmse["RMSE(℃)"], color="tab:orange")
    ax_rmse.set_xlabel("RMSE (℃)")
    ax_rmse.invert_yaxis()
    st.pyplot(fig_rmse)
