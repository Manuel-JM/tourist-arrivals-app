import streamlit as st
import pandas as pd
import numpy as np

st.title("2. Clean the data")

if "raw_df" not in st.session_state:
    st.warning("Run the Dataset page first.")
    st.stop()

SEASON_MAP = {"Dry": 0, "Transition": 1, "Wet": 2}
MONSOON_MAP = {"Amihan": 0, "Transition": 1, "Habagat": 2}

# Philippine border closure to foreign tourists (Mar 2020) until reopening (Feb 2022)
COVID_START, COVID_END = "2020-03-01", "2022-02-01"

# Hard-coded corrections, each with a reason outside the IQR rule itself
CORRECTIONS = {
    "2001-02-01": "Over 30 times the next-highest month in the dataset; treated as a "
                  "data-entry error and re-interpolated from neighbouring months.",
}

if st.button("Run cleaning"):
    df = st.session_state.raw_df.copy()
    rows_before = len(df)

    # 1. Duplicates: keep the first row for each month
    duplicate_rows = df[df.duplicated(subset="date", keep=False)][["date", "arrivals"]]
    duplicates_removed = int(df.duplicated(subset="date").sum())
    df = df.drop_duplicates(subset="date", keep="first").reset_index(drop=True)

    # 2. Missing values: report first, then fill
    missing = df.isna().sum()
    missing = missing[missing > 0]
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].interpolate(limit_direction="both")
    df[["season", "monsoon"]] = df[["season", "monsoon"]].ffill().bfill()

    # 3. Encode the text categories so Spearman and VIF can use them
    df["season"] = df["season"].str.strip()
    df["monsoon"] = df["monsoon"].str.strip()
    df["season_code"] = df["season"].map(SEASON_MAP)
    df["monsoon_code"] = df["monsoon"].map(MONSOON_MAP)
    df["covid_period"] = ((df["date"] >= COVID_START) & (df["date"] <= COVID_END)).astype(int)
    unmapped = sorted(
        set(df.loc[df["season_code"].isna(), "season"])
        | set(df.loc[df["monsoon_code"].isna(), "monsoon"])
    )

    # 4. IQR outlier check on arrivals (flag only)
    q1, q3 = df["arrivals"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flagged = df[(df["arrivals"] < lower) | (df["arrivals"] > upper)][["date", "arrivals"]]

    # 5. Apply documented corrections only
    corrected = []
    for d, reason in CORRECTIONS.items():
        mask = df["date"] == pd.Timestamp(d)
        if mask.any():
            corrected.append({"date": d, "old_value": float(df.loc[mask, "arrivals"].iloc[0]), "reason": reason})
            df.loc[mask, "arrivals"] = np.nan
    df["arrivals"] = df["arrivals"].interpolate(limit_direction="both")
    for c in corrected:
        c["new_value"] = float(df.loc[df["date"] == pd.Timestamp(c["date"]), "arrivals"].iloc[0])

    st.session_state.clean_df = df  # used by every later page
    st.session_state.clean_report = {
        "rows_before": rows_before,
        "rows_after": len(df),
        "duplicates_removed": duplicates_removed,
        "duplicate_rows": duplicate_rows,
        "missing": missing,
        "unmapped": unmapped,
        "bounds": (lower, upper),
        "flagged": flagged,
        "corrected": pd.DataFrame(corrected),
    }

# Displayed outside the button block, so the result stays when you come back
if "clean_report" in st.session_state:
    r = st.session_state.clean_report

    st.subheader("Duplicates")
    st.write(f"Rows before: {r['rows_before']}, after: {r['rows_after']}")
    st.write(f"Duplicates removed: {r['duplicates_removed']}")
    if len(r["duplicate_rows"]):
        st.dataframe(r["duplicate_rows"])

    st.subheader("Missing values (before filling)")
    if len(r["missing"]):
        st.dataframe(r["missing"].rename("missing_count"))
        st.caption("Numeric gaps filled by linear interpolation; season and monsoon by forward fill.")
    else:
        st.write("None")

    st.subheader("Encoding")
    st.write(f"season_code: {SEASON_MAP}")
    st.write(f"monsoon_code: {MONSOON_MAP}")
    st.write(f"covid_period: 1 from {COVID_START} to {COVID_END}, else 0")
    if r["unmapped"]:
        st.warning(f"Labels not in the maps above: {r['unmapped']}")

    st.subheader("Flagged outliers in arrivals (IQR rule)")
    lower, upper = r["bounds"]
    st.write(f"Bounds: {lower:,.0f} to {upper:,.0f}")
    if len(r["flagged"]):
        st.dataframe(r["flagged"])
    else:
        st.write("None flagged")

    st.subheader("Corrections applied")
    if len(r["corrected"]):
        st.dataframe(r["corrected"])
    else:
        st.write("None")

    st.subheader("Arrivals after cleaning")
    st.line_chart(st.session_state.clean_df.set_index("date")["arrivals"])
else:
    st.info('Click "Run cleaning" to process the dataset loaded on the Dataset page.')