import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler

st.title("4. Prevent leakage and prepare sequences")

if "selected_features" not in st.session_state:
    st.warning("Run the Features page first.")
    st.stop()

LOOKBACK = 12  # one year of history feeding each prediction

train_ratio = st.slider("Train split ratio", min_value=0.60, max_value=0.95, value=0.80, step=0.05)
st.caption(
    f"{train_ratio:.0%} train / {1 - train_ratio:.0%} test, split chronologically with no shuffling. "
    "Earliest rows train the model; the most recent rows test it."
)


def make_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i + lookback])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)


if st.button("Run"):
    df = st.session_state.clean_df
    cols = st.session_state.selected_features

    # 1. Chronological split, no shuffling
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    if len(test_df) <= LOOKBACK:
        st.error(f"Test set has only {len(test_df)} rows; it needs more than {LOOKBACK}. Lower the split ratio.")
        st.stop()

    # 2. Scalers fit on TRAIN ONLY
    st.session_state.scaler_X = MinMaxScaler().fit(train_df[cols])
    st.session_state.scaler_y = MinMaxScaler().fit(train_df[["arrivals"]])

    train_X = st.session_state.scaler_X.transform(train_df[cols])
    test_X = st.session_state.scaler_X.transform(test_df[cols])
    train_y = st.session_state.scaler_y.transform(train_df[["arrivals"]])
    test_y = st.session_state.scaler_y.transform(test_df[["arrivals"]])

    # 3. Window each split separately, never across the split point
    st.session_state.X_train_seq, st.session_state.y_train_seq = make_sequences(train_X, train_y, LOOKBACK)
    st.session_state.X_test_seq, st.session_state.y_test_seq = make_sequences(test_X, test_y, LOOKBACK)

    st.session_state.test_df = test_df  # kept for the Evaluate page's baselines

    st.session_state.prepare_report = {
        "train_ratio": train_ratio,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_range": (train_df["date"].min().date(), train_df["date"].max().date()),
        "test_range": (test_df["date"].min().date(), test_df["date"].max().date()),
        "first_test_target": test_df["date"].iloc[LOOKBACK].date(),
        "train_covid_months": int(train_df["covid_period"].sum()) if "covid_period" in cols else None,
        "train_windows": len(st.session_state.X_train_seq),
        "test_windows": len(st.session_state.X_test_seq),
        "x_shape": st.session_state.X_train_seq.shape,
        "lookback": LOOKBACK,
    }

# Displayed outside the button block, so the result stays when you come back
if "prepare_report" in st.session_state:
    r = st.session_state.prepare_report
    st.write(f"Split used: {r['train_ratio']:.0%} train / {1 - r['train_ratio']:.0%} test")
    st.write(f"Train: {r['train_rows']} rows ({r['train_range'][0]} to {r['train_range'][1]})")
    st.write(f"Test: {r['test_rows']} rows ({r['test_range'][0]} to {r['test_range'][1]})")
    st.write(f"First month the model is scored on: {r['first_test_target']} "
             f"(the first {r['lookback']} test months are used as history only)")
    if r["train_covid_months"] is not None:
        st.write(f"COVID months in training data: {r['train_covid_months']}")
    st.write(f"Training input shape (N, L, F): {r['x_shape']}")
    st.success(
        f"{r['train_windows']} training windows, {r['test_windows']} test windows, "
        f"lookback = {r['lookback']}"
    )
else:
    st.info('Choose a split ratio above, then click "Run" to prepare the sequences.')