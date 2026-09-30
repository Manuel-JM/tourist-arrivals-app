import streamlit as st
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

st.title("3. Feature selection")

if "clean_df" not in st.session_state:
    st.warning("Run the Clean page first.")
    st.stop()

CANDIDATES = [
    "quarter", "is_holiday_peak", "season_code", "monsoon_code", "covid_period",
    "temp_mean_c", "temp_min_c", "temp_max_c",
    "rainfall_mm", "rainy_days", "humidity_pct",
    "typhoon_count", "typhoon_max_wind_kt", "storm_signal_days",
    "pm25_ugm3", "wave_height_m",
]
RHO_MIN, ALPHA, VIF_MAX = 0.10, 0.05, 5

# Added after selection. Arrivals can't be Spearman-tested against itself, so it
# skips the filter. Each window covers months t-12 to t-1, so the model only ever
# sees arrivals from BEFORE the month it predicts (no leakage).
EXTRA_INPUTS = ["past_arrivals"]


def compute_vifs(X):
    # Constant added so VIF measures collinearity, not just large average values
    Xc = add_constant(X, has_constant="add")
    return pd.Series(
        [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])],
        index=X.columns,
    )


if st.button("Run Spearman + VIF"):
    df = st.session_state.clean_df
    df["past_arrivals"] = df["arrivals"]  # history input, used only inside windows

    not_found = [c for c in CANDIDATES if c not in df.columns]
    if not_found:
        st.error(f"Columns missing from the cleaned data: {not_found}. Re-run the Clean page.")
        st.stop()

    # Step 1: Spearman filter (|rho| > 0.10 and p < 0.05)
    rows = []
    for col in CANDIDATES:
        rho, p_value = spearmanr(df[col], df["arrivals"])
        rows.append({
            "feature": col,
            "rho": rho,
            "abs_rho": abs(rho),
            "p_value": p_value,
            "kept": bool(abs(rho) > RHO_MIN and p_value < ALPHA),
        })
    spearman = pd.DataFrame(rows).sort_values("abs_rho", ascending=False).reset_index(drop=True)
    kept = spearman.loc[spearman["kept"], "feature"].tolist()

    # Step 2: iterative VIF, dropping the highest until all are below 5
    X = df[kept].copy()
    vif_log = []
    while X.shape[1] > 1:
        vifs = compute_vifs(X)
        if vifs.max() < VIF_MAX:
            break
        drop_col = vifs.idxmax()
        vif_log.append({"step": len(vif_log) + 1, "dropped": drop_col, "vif": float(vifs.max())})
        X = X.drop(columns=[drop_col])

    final_vif = compute_vifs(X) if X.shape[1] > 1 else pd.Series([1.0] * X.shape[1], index=X.columns)

    # Step 3: add past arrivals as a model input, with its VIF shown for transparency
    selected = list(X.columns) + EXTRA_INPUTS
    with_extra_vif = compute_vifs(df[selected])

    st.session_state.selected_features = selected  # used by every later page
    st.session_state.feature_report = {
        "spearman": spearman,
        "passed_spearman": kept,
        "vif_log": pd.DataFrame(vif_log),
        "final_vif": final_vif.rename("vif").reset_index().rename(columns={"index": "feature"}),
        "with_extra_vif": with_extra_vif.rename("vif").reset_index().rename(columns={"index": "feature"}),
    }

# Displayed outside the button block, so the result stays when you come back
if "feature_report" in st.session_state:
    r = st.session_state.feature_report

    st.subheader("Step 1: Spearman filter")
    st.caption(f"Kept if |rho| > {RHO_MIN} and p < {ALPHA}")
    st.dataframe(r["spearman"])
    st.write(f"Passed Spearman ({len(r['passed_spearman'])}): {r['passed_spearman']}")

    st.subheader("Step 2: Iterative VIF")
    st.caption(f"Drop the highest VIF each round until all are below {VIF_MAX}")
    if len(r["vif_log"]):
        st.dataframe(r["vif_log"])
    else:
        st.write("No removals needed")
    st.write("Final VIF values (environmental features):")
    st.dataframe(r["final_vif"])

    st.subheader("Step 3: Add past arrivals as model input")
    st.write(
        "The naive baseline beat the climate-only LSTM, showing that recent arrivals carry "
        "information no climate feature does. past_arrivals gives the LSTM the previous "
        "12 months of arrivals inside each window. It skips the Spearman filter because it "
        "is the target's own history."
    )
    st.write("VIF with past_arrivals included (for reference, not used to drop):")
    st.dataframe(r["with_extra_vif"])

    st.success(f"Selected features: {st.session_state.selected_features}")
else:
    st.info('Click "Run Spearman + VIF" to select features from the cleaned dataset.')