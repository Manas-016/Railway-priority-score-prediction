import pandas as pd

df = pd.read_csv(
    "Datasets/New_data/final_training_data.csv"
)

df["final_priority_score"] = (
    0.6 * df["calculated_priority_score"]
    + 0.4 * df["predicted_priority_score"]
)

score_table = df[
    [
        "original_priority_score",
        "predicted_priority_score",
        "calculated_priority_score",
        "final_priority_score"
    ]
]

print("\nPriority Score Table:\n")
print(score_table.to_string(index=False))

df.to_csv(
    "Datasets/New_data/final_training_data.csv",
    index=False
)

print("\nFinal priority scores saved successfully!")