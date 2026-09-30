import streamlit as st
import numpy as np
import pandas as pd

st.title("6. Evaluate honestly")

if "model" not in st.session_state:
    st.warning("Run the Train page first.")
    st.stop()

SEASONAL_PERIOD = 12  # same calendar month one year back


def score(actual, pred):
    actual, pred = np.asarray(actual).ravel(), np.asarray(pred).ravel()
    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": float(1 - ss_res / ss_tot)}


if st.button("Score on the test set"):
    X_test_seq = st.session_state.X_test_seq
    scaler_y = st.session_state.scaler_y

    pred = np.clip(scaler_y.inverse_transform(st.session_state.model.predict(X_test_seq, verbose=0)).ravel(), 0, None)  # arrivals can't be negative
    actual = scaler_y.inverse_transform(st.session_state.y_test_seq).ravel()

    lookback = X_test_seq.shape[1]
    n_windows = len(X_test_seq)
    if lookback < SEASONAL_PERIOD:
        st.error(f"Seasonal naive needs a lookback of at least {SEASONAL_PERIOD} months; got {lookback}.")
        st.stop()

    # Baselines use the original, unscaled arrivals from the test period.
    # Window i's target is test_df row (i + lookback).
    test_df = st.session_state.test_df.reset_index(drop=True)
    test_arrivals = test_df["arrivals"].to_numpy()
    naive = np.array([test_arrivals[i + lookback - 1] for i in range(n_windows)])
    seasonal_naive = np.array([test_arrivals[i + lookback - SEASONAL_PERIOD] for i in range(n_windows)])

    target_rows = test_df.iloc[lookback:lookback + n_windows]
    covid_mask = target_rows["covid_period"].to_numpy() == 1
    keep = ~covid_mask

    all_months = pd.DataFrame({
        "LSTM": score(actual, pred),
        "Naive": score(actual, naive),
        "Seasonal naive": score(actual, seasonal_naive),
    })
    non_covid = pd.DataFrame({
        "LSTM": score(actual[keep], pred[keep]),
        "Naive": score(actual[keep], naive[keep]),
        "Seasonal naive": score(actual[keep], seasonal_naive[keep]),
    })

    st.session_state.evaluate_report = {
        "all_months": all_months,
        "non_covid": non_covid,
        "n_scored": n_windows,
        "n_covid": int(covid_mask.sum()),
        "range": (target_rows["date"].iloc[0].date(), target_rows["date"].iloc[-1].date()),
        "series": pd.DataFrame({
            "date": target_rows["date"].to_numpy(),
            "Actual": actual,
            "LSTM": pred,
            "Naive": naive,
            "Seasonal naive": seasonal_naive,
        }).set_index("date"),
    }

# Displayed outside the button block, so the result stays when you come back
if "evaluate_report" in st.session_state:
    r = st.session_state.evaluate_report
    st.write(f"Scored on {r['n_scored']} test months ({r['range'][0]} to {r['range'][1]}), "
             f"of which {r['n_covid']} fall in the COVID period.")

    st.subheader("All test months")
    st.dataframe(r["all_months"].style.format("{:,.3f}"))
    best = r["all_months"].loc["MAE"].idxmin()
    st.write(f"Lowest MAE: **{best}**")

    st.subheader("Excluding COVID-period months")
    st.dataframe(r["non_covid"].style.format("{:,.3f}"))

    st.subheader("Actual vs predicted")
    st.line_chart(r["series"])
else:
    st.info('Click "Score on the test set" to evaluate the trained model.')