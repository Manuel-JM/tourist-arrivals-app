import streamlit as st
import pandas as pd

st.title("1. Dataset summary")

if "raw_df" not in st.session_state:
    path = "data/tourist_arrivals.csv"
    with open(path, encoding="utf-8-sig") as f:
        lines = f.readlines()
    header_row = next(
        (i for i, line in enumerate(lines)
         if "date" in [c.strip().strip('"').lower() for c in line.split(",")]),
        None,
    )
    if header_row is None:
        st.error("Couldn't find a header row with a 'date' column. First lines of the file:")
        st.code("".join(lines[:5]))
        st.stop()
    df = pd.read_csv(path, skiprows=header_row, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"])
    st.session_state.raw_df = df.sort_values("date").reset_index(drop=True)

df = st.session_state.raw_df

st.write(f"{len(df)} rows, {len(df.columns)} columns")
st.write(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
st.dataframe(df.head())

st.subheader("Column types")
st.dataframe(df.dtypes.astype(str).rename("dtype"))

expected = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
missing_months = expected.difference(df["date"])
if len(missing_months):
    st.warning(f"Missing months: {[d.date() for d in missing_months]}")
else:
    st.success("No gaps in the monthly sequence.")