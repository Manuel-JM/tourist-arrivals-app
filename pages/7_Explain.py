import streamlit as st
import numpy as np
import pandas as pd
import shap

st.title("7. Explain with SHAP")

if "model" not in st.session_state:
    st.warning("Run the Train page first.")
    st.stop()

N_BACKGROUND, N_CLUSTERS, N_SAMPLES = 50, 10, 500

if st.button("Compute SHAP values"):
    with st.spinner("Computing SHAP values... this can take a few minutes"):
        X_train_seq = st.session_state.X_train_seq
        X_test_seq = st.session_state.X_test_seq
        features = st.session_state.selected_features
        model = st.session_state.model
        scaler_X, scaler_y = st.session_state.scaler_X, st.session_state.scaler_y
        lookback, n_features = X_train_seq.shape[1], X_train_seq.shape[2]
        n_test = len(X_test_seq)

        # DeepExplainer/GradientExplainer fail on LSTM ops in some TF versions,
        # so the model is treated as a black box: KernelExplainer only calls predict().
        def predict_flat(flat_x):
            seq = flat_x.reshape(-1, lookback, n_features)
            return model.predict(seq, batch_size=1024, verbose=0).reshape(-1)

        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train_seq), N_BACKGROUND, replace=False)
        background = X_train_seq[idx].reshape(N_BACKGROUND, -1)
        background_summary = shap.kmeans(background, N_CLUSTERS)

        explainer = shap.KernelExplainer(predict_flat, background_summary)
        sv = explainer.shap_values(X_test_seq.reshape(n_test, -1), nsamples=N_SAMPLES, silent=True)
        sv = np.array(sv).reshape(n_test, lookback, n_features)

        # MinMax scaling is linear, so scaled SHAP values x target range = arrivals
        y_range, y_min = float(scaler_y.data_range_[0]), float(scaler_y.data_min_[0])
        sv_arrivals = sv * y_range
        base = float(np.ravel(explainer.expected_value)[0]) * y_range + y_min

        test_df = st.session_state.test_df.reset_index(drop=True)
        dates = test_df["date"].iloc[lookback:lookback + n_test].dt.strftime("%Y-%m").tolist()

        # Total contribution of each feature to each forecast (summed over the 12 months)
        per_forecast = pd.DataFrame(sv_arrivals.sum(axis=1), index=dates, columns=features)
        global_imp = per_forecast.abs().mean().sort_values(ascending=False)

        # Dependence: value in the month right before the target vs its SHAP value there
        last_values = pd.DataFrame(scaler_X.inverse_transform(X_test_seq[:, -1, :]), index=dates, columns=features)
        last_shap = pd.DataFrame(sv_arrivals[:, -1, :], index=dates, columns=features)

        preds = predict_flat(X_test_seq.reshape(n_test, -1)) * y_range + y_min

        st.session_state.explain_report = {
            "base": base,
            "per_forecast": per_forecast,
            "global": global_imp,
            "last_values": last_values,
            "last_shap": last_shap,
            "preds": pd.Series(preds, index=dates),
        }

# Displayed outside the button block, so the result stays when you come back
if "explain_report" in st.session_state:
    r = st.session_state.explain_report
    features = list(r["global"].index)

    st.subheader("Global feature importance")
    st.caption("Average size of each feature's contribution to a forecast, in arrivals")
    st.bar_chart(r["global"], horizontal=True, sort=False)
    st.dataframe(r["global"].rename("mean |SHAP| (arrivals)").map("{:,.0f}".format))

    st.subheader("One forecast: feature contributions (waterfall)")
    month = st.selectbox("Forecast month", r["per_forecast"].index, index=len(r["per_forecast"]) - 1)
    contrib = r["per_forecast"].loc[month].sort_values(key=abs, ascending=False)
    st.write(
        f"Base value (average prediction): {r['base']:,.0f}  |  "
        f"Sum of contributions: {contrib.sum():+,.0f}  |  "
        f"Model prediction: {r['preds'].loc[month]:,.0f}"
    )
    st.bar_chart(contrib, horizontal=True, sort=False)
    waterfall = pd.DataFrame({
        "contribution": contrib,
        "running_total": r["base"] + contrib.cumsum(),
    })
    st.dataframe(waterfall.map("{:,.0f}".format))

    st.subheader("Dependence plot")
    feat = st.selectbox("Feature", features, index=0)
    dep = pd.DataFrame({"value": r["last_values"][feat], "shap": r["last_shap"][feat]})
    st.scatter_chart(dep, x="value", y="shap", x_label=f"{feat} (month before target)", y_label="SHAP (arrivals)")
else:
    st.info('Click "Compute SHAP values" to explain the trained model\'s predictions.')