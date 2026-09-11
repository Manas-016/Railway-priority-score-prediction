import pandas as pd
import joblib

model = joblib.load("xgboost_priority_model.pkl")
preprocessor = joblib.load("priority_preprocessor.pkl")

df = pd.read_csv("Datasets/final_training_data.csv")

df = df.drop(columns=[
    "department_asset",
    "section_id_asset",
    "location_code_asset",
    "criticality_asset",
    "status_asset"
], errors="ignore")

task_ids = df["task_id"]
asset_ids = df["asset_id"]

 
X = df.drop(columns=[
    "priority_score",
    "task_id",
    "asset_id"
])

X_processed = preprocessor.transform(X)

predictions = model.predict(X_processed)

predictions = predictions.clip(0, 100)


def get_risk_level(score):

    if score >= 75:
        return "HIGH"
    elif score >= 50:
        return "MEDIUM"
    else:
        return "LOW"


def get_priority(score):

    if score >= 75:
        return "URGENT"
    elif score >= 50:
        return "HIGH"
    elif score >= 25:
        return "NORMAL"
    else:
        return "LOW"



results = pd.DataFrame({
    "Task ID": task_ids,
    "Asset ID": asset_ids,
    "Priority Score": predictions.round(2)
})

results["Risk Level"] = results["Priority Score"].apply(get_risk_level)
results["Recommended Priority"] = results["Priority Score"].apply(get_priority)
print(results.shape)
print("Number of rows:", len(results))
print("Number of columns:", len(results.columns))
print("\nFINAL MAINTENANCE PRIORITY RESULTS\n")
print(results.to_string(index=False))
results.to_csv("Datasets/maintenance_priority_predictions.csv", index=False)