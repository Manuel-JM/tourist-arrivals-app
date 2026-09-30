# Philippine Tourist Arrivals Forecast (LSTM + SHAP)

Streamlit app for Laboratory Exercise 1: cleaning, Spearman + VIF feature
selection, leakage-safe preprocessing, LSTM training, evaluation against
naive baselines, SHAP explanations, and an interactive forecast page.

## Run locally
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run Home.py

Run the sidebar pages in order: Dataset, Clean, Features, Prepare, Train,
Evaluate, Explain, Forecast.