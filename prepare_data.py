import pandas as pd


assets = pd.read_csv("Datasets/assets.csv")
tasks = pd.read_csv("Datasets/maintenance_tasks.csv")
defects = pd.read_csv("Datasets/defects.csv")
trains = pd.read_csv("Datasets/train_movements.csv")
tasks = tasks.drop(columns=["priority_score"])

tasks["section_id"] = tasks["section_id"].str.replace("-", "", regex=False)
trains["section_id"] = trains["section_id"].str.replace("-", "", regex=False)

defect_features = defects.groupby("asset_id").agg(

    defect_count=("defect_id", "count"),

    critical_defect_count=(
        "severity",
        lambda x: (x == "Critical").sum()
    ),

    high_defect_count=(
        "severity",
        lambda x: (x == "High").sum()
    ),

    overdue_defect_count=(
        "is_overdue",
        lambda x: (x == "Yes").sum()
    ),

    repeat_defect_count=(
        "repeat_defect",
        lambda x: (x == "Yes").sum()
    )

).reset_index()

print("Task section IDs:")
print(tasks["section_id"].unique()[:20])

print("\nTrain section IDs:")
print(trains["section_id"].unique()[:20])

print("\nMatching section IDs:")
print(
    len(
        set(tasks["section_id"].unique())
        &
        set(trains["section_id"].unique())
    )
)

train_features = trains.groupby("section_id").agg(

    train_count=("movement_id", "count"),

    freight_train_count=(
        "is_freight",
        lambda x: x.sum()
    ),

    high_priority_train_count=(
        "priority",
        lambda x: (x >= 4).sum()
    ),

    critical_train_count=(
        "priority",
        lambda x: (x == 5).sum()
    )

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

df["installation_date"] = pd.to_datetime(
    df["installation_date"]
)

df["last_maintenance_date"] = pd.to_datetime(
    df["last_maintenance_date"]
)

df["due_date"] = pd.to_datetime(
    df["due_date"]
)

df["next_due_date"] = pd.to_datetime(
    df["next_due_date"]
)


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


criticality_map = {
    "Low": 25,
    "Medium": 50,
    "High": 75,
    "Critical": 100
}

df["C"] = df["criticality"].map(criticality_map)


urgency_map = {
    "Low": 25,
    "Medium": 50,
    "High": 75,
    "Critical": 100
}

df["U"] = df["urgency"].map(urgency_map)


impact_map = {
    "Low": 25,
    "Medium": 50,
    "High": 75,
    "Critical": 100
}

df["base_impact"] = df["operational_impact"].map(
    impact_map
)


def normalize(series):

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(0, index=series.index)

    return (
        (series - min_value) /
        (max_value - min_value)
    ) * 100


train_pressure = (
    df["train_count"] * 0.4
    + df["high_priority_train_count"] * 0.3
    + df["critical_train_count"] * 0.3
)

df["train_pressure"] = normalize(train_pressure)


df["I"] = (
    0.6 * df["base_impact"]
    + 0.4 * df["train_pressure"]
)



df["overdue_days"] = (
    -df["days_until_due"]
).clip(lower=0)


overdue_days_score = normalize(
    df["overdue_days"]
)


overdue_defect_score = normalize(
    df["overdue_defect_count"]
)


df["O"] = (
    0.6 * overdue_days_score
    + 0.4 * overdue_defect_score
)


df["priority_score"] = (
    0.30 * df["C"]
    + 0.25 * df["U"]
    + 0.30 * df["I"]
    + 0.15 * df["O"]
)


df["priority_score"] = (
    0.30 * df["C"]
    + 0.25 * df["U"]
    + 0.30 * df["I"]
    + 0.15 * df["O"]
)

df["priority_score"] = df["priority_score"].clip(0, 100)



print("\nPriority Components:")
print(
    df[["C", "U", "I", "O", "priority_score"]].describe()
)

print("\nCorrelation with Priority Score:")

correlation_cols = [
    "C",
    "U",
    "I",
    "O",
    "defect_count",
    "critical_defect_count",
    "overdue_defect_count",
    "repeat_defect_count",
    "train_count",
    "critical_train_count",
    "days_until_due",
    "priority_score"
]

print(
    df[correlation_cols]
    .corr()["priority_score"]
    .sort_values(ascending=False)
)


df = df.drop(columns=[
    "reported_date",
    "due_date",
    "installation_date",
    "last_maintenance_date",
    "next_due_date",
    "C",
    "U",
    "I",
    "O",
    "base_impact",
    "train_pressure",
    "overdue_days"
])



df.to_csv(
    "Datasets/final_training_data.csv",
    index=False
)

print("\nFinal dataset created successfully!")
print("Shape:", df.shape)
print("\nPriority score statistics:")
print(df["priority_score"].describe())