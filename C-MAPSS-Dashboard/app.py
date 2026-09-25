import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np




# Page configuration

st.set_page_config(
    page_title="C-MAPSS RUL Dashboard",
    page_icon="✈️",
    layout="wide"
)
BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = BASE_DIR / "styles.css"

with open(CSS_PATH, "r", encoding="utf-8") as css_file:
    st.markdown(
        f"<style>{css_file.read()}</style>",
        unsafe_allow_html=True
    )



# Title Page

st.markdown(
    """
    <div class="dashboard-header">
        <div class="header-kicker">PREDICTIVE MAINTENANCE SYSTEM</div>
        <div class="header-title">
           C-MAPSS Predictive Maintenance Dashboard
        </div>
        <div class="header-subtitle">
            Remaining Useful Life prediction for turbofan engines using XGBoost, LSTM, and Autoformer.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



# File paths

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"



# dataset

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ["FD001", "FD003"]
)


official_file = DATA_DIR / f"predictions_{dataset_name}.csv"
full_file = DATA_DIR / f"full_predictions_{dataset_name}.csv"

official_data = pd.read_csv(official_file)
full_data = pd.read_csv(full_file)

data = pd.read_csv(full_file)
data.columns = data.columns.str.strip()

if not full_file.exists():
    st.error(f"File not found: {full_file}")
    st.info(
        "Make sure the CSV file is inside the data folder."
    )
    st.stop()




# Check required columns

required_columns = [
    "model",
    "unit_number",
    "time_cycles",
    "true_RUL",
    "predicted_RUL"
]

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    st.error(
        f"Missing columns in the CSV file: {missing_columns}"
    )
    st.stop()


# Sidebar model and engine choices

model_options = sorted(
    data["model"].dropna().unique()
)

selected_model = st.sidebar.selectbox(
    "Prediction Model",
    model_options
)
st.sidebar.caption(
    "Use this option to compare the predictions of XGBoost, LSTM, and Autoformer."
)


filtered_model_data = data[
    data["model"] == selected_model
].copy()

engine_options = sorted(
    filtered_model_data["unit_number"].unique()
)

selected_engine = st.sidebar.selectbox(
    "Select Engine",
    engine_options
)



# Filter selected engine

engine_data = filtered_model_data[
    filtered_model_data["unit_number"] == selected_engine
].copy()

engine_data = engine_data.sort_values(
    "time_cycles"
)
selected_record = engine_data.tail(1)

latest_row = engine_data.iloc[-1]

predicted_rul = float(
    latest_row["predicted_RUL"]
)

actual_rul = float(
    latest_row["true_RUL"]
)

last_cycle = int(
    latest_row["time_cycles"]
)

prediction_error = predicted_rul - actual_rul
absolute_error = abs(prediction_error)



# Health classification

st.subheader("Engine Health Status")

def classify_condition(rul):
    if rul > 60:
        return "Healthy"
    elif rul > 30:
        return "Warning"
    else:
        return "Critical"


condition = classify_condition(predicted_rul)



# Main metrics

st.subheader("Final RUL Prediction at the Last Operating Cycle")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Dataset",
        dataset_name
    )

with col2:
    st.metric(
        "Engine ID",
        selected_engine
    )

with col3:
    st.metric(
        "Predicted RUL",
        f"{predicted_rul:.2f} cycles"
    )

with col4:
    st.metric(
        "Last Cycle",
        last_cycle
    )


# Condition message

if condition == "Healthy":
    st.success(f"Engine condition: {condition}")
elif condition == "Warning":
    st.warning(f"Engine condition: {condition}")
else:
    st.error(f"Engine condition: {condition}")



# Prediction details

st.subheader("Prediction Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Actual RUL",
        f"{actual_rul:.2f} cycles"
    )

with col2:
    st.metric(
        "Prediction Error",
        f"{prediction_error:.2f} cycles"
    )

with col3:
    st.metric(
        "Absolute Error",
        f"{absolute_error:.2f} cycles"
    )



# RUL graph

engine_data = full_data[
    (full_data["model"] == selected_model) &
    (full_data["unit_number"] == selected_engine)
].copy()

engine_data = engine_data.sort_values(
    "time_cycles"
)


st.subheader("Remaining Useful Life Trend")
st.caption(
    "Comparison between actual and predicted RUL across the available "
    "operating cycles."
)


plot_data = engine_data.sort_values(
    "time_cycles"
).copy()

fig = px.line(
    plot_data,
    x="time_cycles",
    y=["true_RUL", "predicted_RUL"],
    markers=True,
    labels={
        "time_cycles": "Operating Cycle",
        "value": "RUL (Cycles)",
        "variable": "RUL Type"
    },
    title=(
        f"{dataset_name} | Engine {selected_engine} | "
        f"{selected_model}"
    ),
    color_discrete_map={
        "true_RUL": "#2563EB",
        "predicted_RUL": "#F59E0B"
    }
)

fig.update_traces(
    line={"width": 3},
    marker={"size": 5}
)

fig.update_layout(
    height=450,
    hovermode="x unified",
    legend_title_text="",
    xaxis_title="Operating Cycle",
    yaxis_title="Remaining Useful Life (Cycles)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)




# Model comparison

st.subheader("Model Comparison")


def cmapss_score(y_true, y_pred):
    """
    NASA C-MAPSS asymmetric scoring function.

    Lower score is better.
    """
    errors = y_pred - y_true

    penalties = (
        (errors < 0) * (np.exp(-errors / 13) - 1)
        + (errors >= 0) * (np.exp(errors / 10) - 1)
    )

    return penalties.sum()


comparison = official_data.copy()


comparison["error"] = (
    comparison["predicted_RUL"] -
    comparison["true_RUL"]
)

comparison["absolute_error"] = (
    comparison["error"].abs()
)

comparison["squared_error"] = (
    comparison["error"] ** 2
)


summary_rows = []

for model_name, model_data in comparison.groupby("model"):

    y_true = model_data["true_RUL"].to_numpy()
    y_pred = model_data["predicted_RUL"].to_numpy()

    mae = model_data["absolute_error"].mean()

    rmse = (
        model_data["squared_error"].mean()
    ) ** 0.5

    nasa_score = cmapss_score(
        y_true,
        y_pred
    )

    summary_rows.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "NASA Score": nasa_score
    })


model_summary = pd.DataFrame(summary_rows)

model_summary["MAE"] = model_summary["MAE"].round(2)
model_summary["RMSE"] = model_summary["RMSE"].round(2)
model_summary["NASA Score"] = (
    model_summary["NASA Score"].round(2)
)

st.dataframe(
    model_summary,
    use_container_width=True,
    hide_index=True
)
best_mae_model = model_summary.loc[
    model_summary["MAE"].idxmin(),
    "Model"
]

best_rmse_model = model_summary.loc[
    model_summary["RMSE"].idxmin(),
    "Model"
]

best_nasa_model = model_summary.loc[
    model_summary["NASA Score"].idxmin(),
    "Model"
]

st.success(
    f"Best model according to MAE: {best_mae_model}"
)

st.info(
    f"Best model according to RMSE: {best_rmse_model}"
)

st.warning(
    f"Best model according to NASA Score: {best_nasa_model}"
)




# data

with st.expander("View Selected Prediction Record"):
  st.dataframe(
    selected_record,
    use_container_width=True,
    hide_index=True
)


st.caption(
    "Healthy, Warning, and Critical thresholds are project-defined "
    "for dashboard visualization."
)
