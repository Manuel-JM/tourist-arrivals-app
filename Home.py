import streamlit as st

st.set_page_config(page_title="Tourist Arrivals Forecast", layout="wide")
st.title("Philippine Tourist Arrivals: Forecasting Lab")
st.write(
    "Use the pages in the sidebar, in order: Dataset, Clean, Features, "
    "Prepare, Train, Evaluate, Explain, Forecast. Each page depends on "
    "the one before it having been run at least once in this session."
)