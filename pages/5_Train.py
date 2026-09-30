import streamlit as st
import itertools
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

st.title("5. Train and tune the LSTM")

if "X_train_seq" not in st.session_state:
    st.warning("Run the Prepare page first.")
    st.stop()

PARAM_GRID = {"units": [32, 64], "dropout": [0.1, 0.3], "batch_size": [16, 32]}
SEED = 42  # same seed for every candidate, so results are repeatable


def build_model(lookback, n_features, units, dropout):
    model = Sequential([
        Input(shape=(lookback, n_features)),
        LSTM(units),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


if st.button("Train & tune"):
    X = st.session_state.X_train_seq
    y = st.session_state.y_train_seq
    lookback, n_features = X.shape[1], X.shape[2]

    combos = list(itertools.product(*PARAM_GRID.values()))
    progress = st.progress(0.0, text="Starting...")
    results = []
    best = {"val_loss": float("inf")}

    # Step 1: hyperparameter search, scored on validation data only
    for k, (units, dropout, batch_size) in enumerate(combos):
        progress.progress(
            k / (len(combos) + 1),
            text=f"Trying {k + 1}/{len(combos)}: units={units}, dropout={dropout}, batch={batch_size}",
        )
        tf.keras.utils.set_random_seed(SEED)
        model = build_model(lookback, n_features, units, dropout)

        # validation_split takes the LAST 15% of training windows (chronological).
        # The test set is never used here.
        hist = model.fit(
            X, y,
            validation_split=0.15,
            epochs=100,
            batch_size=batch_size,
            callbacks=[EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
            verbose=0,
        )

        val_loss = float(min(hist.history["val_loss"]))
        results.append({
            "units": units,
            "dropout": dropout,
            "batch_size": batch_size,
            "epochs_run": len(hist.history["loss"]),
            "best_val_loss": val_loss,
        })
        if val_loss < best["val_loss"]:
            best = {
                "val_loss": val_loss,
                "params": {"units": units, "dropout": dropout, "batch_size": batch_size},
                "history": hist.history,
            }

    # Step 2: refit the winning settings on ALL training windows.
    # The validation tail holds every COVID month in the training data, so the
    # tuned model never trained on them. The refit fixes that. Validation loss
    # can't be used to stop here, so it stops when training loss stops improving.
    progress.progress(len(combos) / (len(combos) + 1), text="Refitting best settings on all training windows...")
    p = best["params"]
    tf.keras.utils.set_random_seed(SEED)
    final_model = build_model(lookback, n_features, p["units"], p["dropout"])
    refit = final_model.fit(
        X, y,
        epochs=100,
        batch_size=p["batch_size"],
        callbacks=[EarlyStopping(monitor="loss", patience=8, restore_best_weights=True)],
        verbose=0,
    )
    progress.progress(1.0, text="Done")

    st.session_state.model = final_model  # used by every later page
    st.session_state.train_report = {
        "best_params": p,
        "val_loss": best["val_loss"],
        "grid": pd.DataFrame(results).sort_values("best_val_loss").reset_index(drop=True),
        "loss_history": best["history"]["loss"],
        "val_loss_history": best["history"]["val_loss"],
        "refit_loss_history": refit.history["loss"],
    }

# Displayed outside the button block, so the result stays when you come back
if "train_report" in st.session_state:
    r = st.session_state.train_report
    st.success(f"Best hyperparameters: {r['best_params']} (val_loss = {r['val_loss']:.4f})")

    st.subheader("Step 1: All candidates (scored on validation data only)")
    st.dataframe(r["grid"])

    st.subheader("Tuning run: training vs validation loss (best candidate)")
    st.line_chart(
        pd.DataFrame({"loss": r["loss_history"], "val_loss": r["val_loss_history"]}),
        x_label="epoch",
        y_label="MSE (scaled)",
    )

    st.subheader("Step 2: Final refit on all training windows")
    st.write(f"Epochs run: {len(r['refit_loss_history'])}")
    st.line_chart(
        pd.DataFrame({"loss": r["refit_loss_history"]}),
        x_label="epoch",
        y_label="MSE (scaled)",
    )
else:
    st.info('Click "Train & tune" to fit the LSTM on the sequences from the Prepare page.')