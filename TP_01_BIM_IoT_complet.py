# BIM Risk Prediction — Civil Engineering Projects
# Predicts project risk level (Low / Medium / High) using IoT sensors,
# drone surveillance and BIM data.
# Dataset : bim_ai_civil_engineering_dataset.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.patches import Patch

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

PALETTE = {"Low": "#2E86C1", "Medium": "#F39C12", "High": "#C0392B"}
ORDER   = ["Low", "Medium", "High"]
SEED    = 42

# ------------------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------------------

df = pd.read_csv("bim_ai_civil_engineering_dataset.csv")

print(f"Dataset: {df.shape[0]} projects x {df.shape[1]} features")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nRisk level distribution:\n{df['Risk_Level'].value_counts()}")
print(f"\nProject types:\n{df['Project_Type'].value_counts()}")

# ------------------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ------------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.countplot(data=df, x="Risk_Level", palette=PALETTE, order=ORDER, ax=axes[0])
axes[0].set_title("Risk Level Distribution")

risk_by_type = pd.crosstab(df["Project_Type"], df["Risk_Level"],
                            normalize="index")[ORDER] * 100
risk_by_type.plot(kind="bar",
                  color=[PALETTE[k] for k in ORDER],
                  edgecolor="white", ax=axes[1])
axes[1].set_title("Risk Level by Project Type (%)")
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend(title="Risk")
plt.tight_layout()
plt.savefig("fig_overview.png", dpi=150)
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (col, label) in zip(axes, [
    ("Cost_Overrun",      "Cost Overrun ($)"),
    ("Schedule_Deviation","Schedule Deviation (days)"),
    ("Safety_Risk_Score", "Safety Risk Score"),
]):
    sns.boxplot(data=df, x="Risk_Level", y=col, order=ORDER, palette=PALETTE, ax=ax)
    ax.set_title(label)
    ax.set_xlabel("Risk Level")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
plt.suptitle("Financial, Planning & Safety Indicators vs Risk Level",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_indicators.png", dpi=150)
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (col, label) in zip(axes, [
    ("Vibration_Level",       "Vibration Level"),
    ("Crack_Width",           "Crack Width (mm)"),
    ("Load_Bearing_Capacity", "Load Bearing Capacity"),
]):
    sns.boxplot(data=df, x="Risk_Level", y=col, order=ORDER, palette=PALETTE, ax=ax)
    ax.set_title(label)
    ax.set_xlabel("Risk Level")
plt.suptitle("IoT Structural Sensors (BIM) vs Risk Level",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_iot_sensors.png", dpi=150)
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (col, label) in zip(axes, [
    ("Image_Analysis_Score",  "Image Analysis Score"),
    ("Completion_Percentage", "Completion (%)"),
    ("Anomaly_Detected",      "Anomaly Detected"),
]):
    sns.boxplot(data=df, x="Risk_Level", y=col, order=ORDER, palette=PALETTE, ax=ax)
    ax.set_title(label)
    ax.set_xlabel("Risk Level")
plt.suptitle("Drone Surveillance vs Risk Level",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_drone.png", dpi=150)
plt.show()


plt.figure(figsize=(12, 9))
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
            cmap="coolwarm", center=0, linewidths=0.3, annot=False)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("fig_correlation.png", dpi=150)
plt.show()

df_tmp = df.copy()
df_tmp["risk_enc"] = df_tmp["Risk_Level"].map({"Low": 0, "Medium": 1, "High": 2})
print("\nTop correlations with Risk_Level:")
print(df.select_dtypes(include=np.number)
       .corrwith(df_tmp["risk_enc"])
       .abs().sort_values(ascending=False)
       .head(10).round(3))

# ------------------------------------------------------------------------------
# 3. Preprocessing
# ------------------------------------------------------------------------------

df_ml = (df.drop(columns=["Project_ID", "Start_Date", "End_Date", "Location"])
           .copy())

df_ml["Risk_Level"] = df_ml["Risk_Level"].map({"Low": 0, "Medium": 1, "High": 2})
df_ml = pd.get_dummies(df_ml, columns=["Project_Type", "Weather_Condition"],
                        drop_first=False)

X = df_ml.drop("Risk_Level", axis=1)
y = df_ml["Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
print(f"Features: {list(X.columns)}")

# ------------------------------------------------------------------------------
# 4. Training — Random Forest
# ------------------------------------------------------------------------------

model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ------------------------------------------------------------------------------
# 5. Evaluation
# ------------------------------------------------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=ORDER))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=ORDER).plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix (counts)")

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, normalize="true").round(2),
                       display_labels=ORDER).plot(ax=axes[1], colorbar=False, cmap="Greens")
axes[1].set_title("Confusion Matrix (normalized)")
plt.suptitle("Random Forest — Evaluation", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_confusion.png", dpi=150)
plt.show()

# ------------------------------------------------------------------------------
# 6. Feature Importance
# ------------------------------------------------------------------------------

CATEGORY_COLORS = {
    "Financial":   "#e74c3c",
    "Planning":    "#e67e22",
    "IoT":         "#2980b9",
    "Environment": "#27ae60",
    "Drone":       "#8e44ad",
    "Other":       "#7f8c8d",
}

def categorize(col):
    if col in ["Cost_Overrun", "Planned_Cost", "Actual_Cost"]:          return "Financial"
    if col in ["Schedule_Deviation", "Planned_Duration","Actual_Duration"]: return "Planning"
    if col in ["Vibration_Level", "Crack_Width", "Load_Bearing_Capacity"]:  return "IoT"
    if col in ["Temperature", "Humidity", "Air_Quality_Index"] or col.startswith("Weather"): return "Environment"
    if col in ["Image_Analysis_Score", "Anomaly_Detected",
               "Completion_Percentage", "Safety_Risk_Score"]:           return "Drone"
    return "Other"

importances = pd.Series(model.feature_importances_, index=X.columns)
top10       = importances.nlargest(10).sort_values()

plt.figure(figsize=(10, 6))
plt.barh(top10.index, top10.values,
         color=[CATEGORY_COLORS[categorize(c)] for c in top10.index],
         edgecolor="white")
plt.xlabel("Importance (Gini)")
plt.title("Top 10 Feature Importances — Random Forest")
plt.legend(handles=[Patch(facecolor=v, label=k) for k, v in CATEGORY_COLORS.items()
                    if k != "Other"],
           loc="lower right", title="BIM Category")
plt.tight_layout()
plt.savefig("fig_feature_importance.png", dpi=150)
plt.show()

print("\nTop 10 features:")
print(importances.nlargest(10).round(4).to_string())

# ------------------------------------------------------------------------------
# 7. Inference — new project
# ------------------------------------------------------------------------------

sample = {
    "Planned_Cost": 5_000_000, "Actual_Cost": 7_500_000, "Cost_Overrun": 2_500_000,
    "Planned_Duration": 365,   "Actual_Duration": 480,   "Schedule_Deviation": 115,
    "Vibration_Level": 2.1,    "Crack_Width": 3.5,       "Load_Bearing_Capacity": 300,
    "Temperature": 35,         "Humidity": 70,            "Air_Quality_Index": 150,
    "Energy_Consumption": 30_000, "Material_Usage": 400, "Labor_Hours": 8_000,
    "Equipment_Utilization": 75,  "Accident_Count": 4,   "Safety_Risk_Score": 7.5,
    "Image_Analysis_Score": 60,   "Anomaly_Detected": 1, "Completion_Percentage": 55,
}

new_proj = pd.DataFrame([sample])
for col in X.columns:
    if col not in new_proj.columns:
        new_proj[col] = 0
if "Project_Type_Tunnel"        in X.columns: new_proj["Project_Type_Tunnel"]        = 1
if "Weather_Condition_Cloudy"   in X.columns: new_proj["Weather_Condition_Cloudy"]   = 1
new_proj = new_proj[X.columns]

pred  = model.predict(new_proj)[0]
proba = model.predict_proba(new_proj)[0]
label = {0: "Low", 1: "Medium", 2: "High"}[pred]

print(f"\nPredicted Risk Level : {label}")
print("Class probabilities  :")
for i, lbl in enumerate(ORDER):
    print(f"  {lbl:<6} : {proba[i]*100:.1f}%")