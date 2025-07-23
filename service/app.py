from __future__ import annotations
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import streamlit as st

# ───────────────────────────── Page Config ──────────────────────────────
st.set_page_config(
    page_title="RKSI Temp Model Dashboard",
    page_icon="🌡️",
    layout="wide",
)

# ─────────────────────────────── Data IO ───────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "results.csv"  # required

@st.cache_data
def load_metrics(path: Path) -> pd.DataFrame:
    """Read CSV with utf‑8 / cp949 fallback & cache the result."""
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

# ────────────────────────── Derived Statistics ─────────────────────────
baseline_row = df.loc[df["model"].str.lower() == "linear"].squeeze()  # baseline for Δ
best_mae_idx = df["MAE(℃)"].idxmin()
best_rmse_idx = df["RMSE(℃)"].idxmin()

# ─────────────────────────────── Header ────────────────────────────────
now_kst = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%B %d, %Y")
st.title("📊 Temperature Prediction Model Comparison")
st.caption(f"Data as of {now_kst}")

# ─────────────────────────────── Layout ────────────────────────────────
OVERVIEW_TAB, DETAILS_TAB = st.tabs(["🔍 Overview", "📑 Details"])

# ───────────────────────────── Overview Tab ────────────────────────────
with OVERVIEW_TAB:
    # ---- Metric cards --------------------------------------------------
    cols = st.columns(len(df))
    for col, (_, row) in zip(cols, df.iterrows()):
        delta_val = row["MAE(℃)"] - baseline_row["MAE(℃)"]
        delta_str = f"{delta_val:+.3f} ℃" if not pd.isna(delta_val) else "‑"
        with col:
            st.metric(
                label=row["model"].upper(),
                value=f"{row['MAE(℃)']:.3f} ℃",
                delta=delta_str,
                help=f"RMSE: {row['RMSE(℃)']:.3f} ℃",
            )
            st.markdown("---")

    # ---- Interactive scatter ------------------------------------------
    scatter_fig = px.scatter(
        df,
        x="MAE(℃)",
        y="RMSE(℃)",
        text="model",
        size_max=16,
        hover_data={"MAE(℃)":":.3f", "RMSE(℃)":":.3f"},
    )
    scatter_fig.update_traces(textposition="top center")
    scatter_fig.update_layout(height=450, dragmode="pan")
    st.subheader("MAE vs. RMSE (lower‑left is better)")
    st.plotly_chart(scatter_fig, use_container_width=True)

# ───────────────────────────── Details Tab ─────────────────────────────
with DETAILS_TAB:
    metric_choice = st.radio("Sort by:", ("MAE", "RMSE"), horizontal=True)
    sort_key = "MAE(℃)" if metric_choice == "MAE" else "RMSE(℃)"
    df_sorted = df.sort_values(sort_key, kind="mergesort")  # stable sort for predictability

    # ---- Styled table --------------------------------------------------
    st.dataframe(
        df_sorted.style
        .background_gradient(cmap="PuBu_r", subset=[sort_key])
        .format({"MAE(℃)": "{:.3f}", "RMSE(℃)": "{:.3f}"}),
        use_container_width=True,
    )

    # ---- Horizontal bar chart -----------------------------------------
    bar_fig = px.bar(
        df_sorted,
        x=sort_key,
        y="model",
        orientation="h",
        text_auto=".3f",
        height=350,
        color="model",
    )
    bar_fig.update_layout(showlegend=False, yaxis=dict(categoryorder="total ascending"))
    st.subheader(f"{metric_choice} by Model (sorted)")
    st.plotly_chart(bar_fig, use_container_width=True)

    # ---- Optional: Prediction vs. Actual plot (if files exist) ---------
    y_true_path = BASE_DIR / "y_true.csv"
    if y_true_path.exists():
        best_model_name = df.loc[best_mae_idx, "model"].lower()
        y_pred_path = BASE_DIR / f"y_pred_{best_model_name}.csv"
        if y_pred_path.exists():
            y_true = pd.read_csv(y_true_path, parse_dates=["date"])
            y_pred = pd.read_csv(y_pred_path, parse_dates=["date"])
            tmp = y_true.merge(y_pred, on="date", suffixes=("_true", "_pred"))
            line_fig = px.line(
                tmp,
                x="date",
                y=[col for col in tmp.columns if col.startswith("temp_")],
                labels={"value": "Temperature (℃)", "variable": ""},
            )
            st.subheader(f"📈 Best Model ({best_model_name}) – Actual vs Predicted")
            st.plotly_chart(line_fig, use_container_width=True)