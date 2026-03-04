# BIM Risk Prediction

Predicts civil engineering project risk level (Low / Medium / High)
using IoT sensor data, drone surveillance and BIM project indicators.

## Stack
Python · Pandas · Scikit-learn · Seaborn · Matplotlib

## Dataset
1 000 projects · 28 features · 0 missing values
Source : bim_ai_civil_engineering_dataset.csv

## Model
Random Forest Classifier — 100 estimators
Accuracy : 92% | F1 High : 0.95 | F1 Low : 0.87

## Run
pip install pandas numpy matplotlib seaborn scikit-learn
python bim_risk_prediction.py
