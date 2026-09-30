import streamlit as st
import numpy as np
import pandas as pd

st.title("8. Forecast")

if "model" not in st.session_state:
    st.warning("Run the Train page first.")
    st.stop()

cols = st.session_state.selected_features
lookback = st.session_state.X_train_seq.shape[1]
df = st.session_state.clean_df
split_idx = len(df) - len(st.session_state.test_df)

source = st.radio(
    "Start from",
    ["Last training window", "Latest 12 months in the dataset"],
    horizontal=True,
)

if source == "Last training window":
    # Same rows as X_train_seq[-1]; its target is the last training month
    window = df.iloc[split_idx - lookback - 1: split_idx - 1]
    target_row = df.iloc[split_idx - 1]
    target_label = target_row["date"].strftime("%Y-%m")
    actual = float(target_row["arrivals"])
else:
    window = df.iloc[-lookback:]
    target_label = (df["date"].iloc[-1] + pd.DateOffset(months=1)).strftime("%Y-%m")
    actual = None

default_df = window[cols].copy()
default_df.index = window["date"].dt.strftime("%Y-%m")

st.write(f"Edit the {lookback} months of readings below, then forecast **{target_label}**:")
st.caption("Keep the column order as is; the model reads the columns in this exact order.")
edited = st.data_editor(default_df, num_rows="fixed", key=f"editor_{source}")

if st.button(f"Forecast {target_label}"):
    X = st.session_state.scaler_X.transform(edited[cols]).reshape(1, lookback, len(cols))
    pred_scaled = st.session_state.model.predict(X, verbose=0)
    pred = float(np.clip(st.session_state.scaler_y.inverse_transform(pred_scaled)[0, 0], 0, None))
    st.session_state.forecast_report = {
        "source": source,
        "target": target_label,
        "pred": pred,
        "actual": actual,
    }

# Displayed outside the button block, so the result stays when you come back
r = st.session_state.get("forecast_report")
if r and r["source"] == source:
    st.success(f"Predicted arrivals for {r['target']}: {r['pred']:,.0f}")
    if r["actual"] is not None:
        err = r["pred"] - r["actual"]
        st.write(f"Actual arrivals that month: {r['actual']:,.0f} (error: {err:+,.0f})")
        st.caption("This compares against the unedited real value; editing the table changes the prediction only.")
else:
    st.info('Edit the table above, then click the forecast button.')