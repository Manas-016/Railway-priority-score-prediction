import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor



df = pd.read_csv("Datasets/final_training_data.csv")

df = df.drop(columns=[
    "department_asset",
    "section_id_asset",
    "location_code_asset",
    "criticality_asset",
    "status_asset"
], errors="ignore")



X = df.drop(columns=[
    "priority_score",
    "task_id",
    "asset_id"
])

y = df["priority_score"]



categorical_cols = [
    "department",
    "task_type",
    "section_id",
    "location_code",
    "asset_type",
    "maintenance_type",
    "status",
    "criticality",
    "urgency",
    "operational_impact",
    "required_block_type",
    "can_be_rescheduled"
]

numeric_cols = [
    col for col in X.columns
    if col not in categorical_cols
]


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])



model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)



kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

mae_scores = []
rmse_scores = []
r2_scores = []


for fold, (train_index, test_index) in enumerate(
    kf.split(X), start=1
):

    print(f"\nFOLD - {fold} ")

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)


    fold_model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )


    fold_model.fit(
        X_train_processed,
        y_train
    )


    predictions = fold_model.predict(
        X_test_processed
    )


    
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )


    mae_scores.append(mae)
    rmse_scores.append(rmse)
    r2_scores.append(r2)


    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


print("\n\n")
print("      CROSS-VALIDATION RESULTS")
print("")

print("\nMean and Std of MAE")
print(f"Mean : {np.mean(mae_scores):.4f}")
print(f"Std  : {np.std(mae_scores):.4f}")

print("\nMean and Std of RMSE")
print(f"Mean : {np.mean(rmse_scores):.4f}")
print(f"Std  : {np.std(rmse_scores):.4f}")

print("\nMean and Std of R²")
print(f"Mean : {np.mean(r2_scores):.4f}")
print(f"Std  : {np.std(r2_scores):.4f}")



results = pd.DataFrame({
    "Fold": [1, 2, 3, 4, 5],
    "MAE": mae_scores,
    "RMSE": rmse_scores,
    "R2": r2_scores
})

print("\n\nFold-wise results:")
print(results)


results.to_csv(
    "Datasets/cross_validation_results.csv",
    index=False
)

print("\nValidation results saved to:")
print("cross_validation_results.csv")