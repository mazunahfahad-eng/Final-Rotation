# Final-Rotation
**Remaining Useful Life (RUL) Prediction of Turbofan Engines**

*An Exploratory and Predictive Machine Learning and Deep Learning Study of Engine Degradation Using NASA C-MAPSS Data*
## 1. Problem Statement
Turbofan engines are critical, high-cost assets whose unexpected failure causes safety risks and costly downtime. Traditional maintenance is either reactive or based on fixed schedules that replace components too early or too late. As engines operate, their sensors detect subtle signs of degradation that are challenging to interpret manually. This project uses the NASA C-MAPSS sensor data to estimate the Remaining Useful Life (RUL) of an engine, allowing the trade-off between the cost of scheduled maintenance and the risk of unplanned downtime to be weighed, and enabling a shift from reactive to preventive maintenance.

## 2. Objective
The project aims to develop a predictive model that estimates the Remaining Useful Life (RUL) of turbofan engines using NASA C-MAPSS sensor data, enabling early detection of engine degradation and supporting predictive maintenance decisions.

## 3. Dataset
The project will use the C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset from NASA (CMAPSS Jet Engine Simulated Data, n.d.), available on the NASA Open Data Portal. The dataset consists of multi-variate sensor time series data from simulated turbofan engines run to failure. Each record contains an engine unit identifier, the operating cycle, three operational settings, and 21 sensor measurements  
  Dataset Source:  CMAPSS Jet Engine Simulated Data (https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
## 4. Proposed models

| Model | Type | Key Approach | Feature Input |
| :--- | :--- | :--- | :--- |
| **XGBoost (baseline)** | Gradient-boosted decision trees | Sequential boosting with early stopping and hyperparameter tuning; a strong tabular benchmark | Lagged sensor readings, rolling statistics, degradation-trend features |
| **LSTM (baseline)** | Recurrent Deep Learning | Baseline recurrent model; sequential gated memory cells over fixed lookback windows to capture temporal degradation dynamics | Scaled multivariate sensor sequences (sliding windows) |
| **Autoformer** | Transformer-based deep learning | Decomposition and autocorrelation mechanisms to capture long-term degradation patterns | Scaled multivariate sensor sequences (sliding windows) |

## 5. Experiment Design
Data Preparation
- Preprocess the C-MAPSS sensor data by removing constant or uninformative sensors, normalizing the selected measurements, and computing piecewise-linear RUL targets.
- Split the engines by unit number into training and validation sets to prevent data leakage between cycles from the same engine.

Models 
- XGBoost as the tuned gradient-boosting benchmark (with hyperparameter tuning)
- LSTM as the baseline deep learning model using sliding-window sensor inputs.
- Autoformer as an advanced Transformer-based model for capturing long-term degradation patterns.

Evaluation Metrics
- NASA C-MAPSS scoring function (asymmetric penalty, weighted more heavily against late predictions) on the official test split.
- RMSE and MAE for interpretability.
  
## 6. Expected outcome
By the end of this project, we expect to deliver a predictive maintenance prototype that estimates the Remaining Useful Life (RUL) of turbofan engines and assigns healthy, warning, or critical condition levels. The project is expected to produce:
- An exploratory analysis of engine run-to-failure cycles and sensor degradation trends.
- A cleaned and model-ready dataset with selected sensor features and suitable RUL targets.
- A comparison of XGBoost, LSTM as the baseline model, and Autoformer using MAE, RMSE, and the NASA scoring function.
- A simple dashboard displaying the predicted RUL and the corresponding engine condition.
ٍ
## References
[1] R. Lin, Y. Yu, H. Wang, C. Che, and X. Ni, “Remaining useful life prediction in prognostics using multi-scale sequence and Long Short-Term Memory network⋆,” Journal of Computational Science, vol. 57, p. 101508, Jan. 2022.

[2] H. Tian, L. Yang, and B. Ju, “Spatial Correlation and Temporal Attention-Based LSTM for Remaining Useful Life Prediction of Turbofan Engine,” Measurement, vol. 214, Art. 112816, 2023.

[3] H. Wu, J. Xu, J. Wang, and M. Long, “Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting,” arXiv (Cornell University), Dec. 2021

[4] A. Gupta, “Auditing Operating-Condition Normalization in C-MAPSS Remaining Useful Life Evaluation.” Open Engineering Inc, Sep. 04, 2026.

[5] S. Deng and J. Zhou, “Prediction of Remaining Useful Life of Aero-Engines Based on CNN-LSTM-Attention,” International Journal of Computational Intelligence Systems, vol. 17, Art. 232, 2024.

[6] A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage propagation modeling for aircraft engine run-to-failure simulation," 2008 International Conference on Prognostics and Health Management, Denver, CO, USA, 2008.

