import pandas as pd

assets = pd.read_csv('Datasets/assets.csv')
tasks = pd.read_csv("Datasets/maintenance_tasks.csv")
defects = pd.read_csv("Datasets/defects.csv")
trains = pd.read_csv("Datasets/train_movements.csv")


defect_features = defects.groupby("asset_id").agg(
    defect_count=("defect_id", "count"),
    critical_defect_count=("severity", lambda x: (x == "Critical").sum()),
    high_defect_count=("severity", lambda x: (x == "High").sum()),
    overdue_defect_count=("is_overdue", lambda x: (x == "Yes").sum()),
    repeat_defect_count=("repeat_defect", lambda x: (x == "Yes").sum())
).reset_index()


train_features = trains.groupby("section_id").agg(
    train_count=("movement_id", "count"),
    freight_train_count=("is_freight", lambda x: (x == "Yes").sum()),
    high_priority_train_count=("priority", lambda x: (x == "High").sum()),
    critical_train_count=("priority", lambda x: (x == "Critical").sum())
).reset_index()

df = tasks.merge(
    assets,
    on="asset_id",
    how="left",
    suffixes=("", "_asset")
)

df = df.merge(
    defect_features,
    on="asset_id",
    how="left"
)

df = df.merge(
    train_features,
    on="section_id",
    how="left"
)


today = pd.Timestamp("2026-09-10")

df["installation_date"] = pd.to_datetime(df["installation_date"])
df["last_maintenance_date"] = pd.to_datetime(df["last_maintenance_date"])
df["due_date"] = pd.to_datetime(df["due_date"])

df["asset_age_years"] = (
    (today - df["installation_date"]).dt.days / 365.25
)

df["days_since_last_maintenance"] = (
    today - df["last_maintenance_date"]
).dt.days

df["days_until_due"] = (
    df["due_date"] - today
).dt.days


df = df.fillna({
    "defect_count": 0,
    "critical_defect_count": 0,
    "high_defect_count": 0,
    "overdue_defect_count": 0,
    "repeat_defect_count": 0,
    "train_count": 0,
    "freight_train_count": 0,
    "high_priority_train_count": 0,
    "critical_train_count": 0
})

df = df.drop(columns=[
    'reported_date',
    'due_date',
    'installation_date',
    'last_maintenance_date',
    'next_due_date'
])

df.to_csv("Datasets/final_training_data.csv",index=False)