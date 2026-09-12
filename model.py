import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



file_path = "Datasets/New_data/final_training_data.csv"

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)



df = df.drop(columns=[
    "department_asset",
    "section_id_asset",
    "location_code_asset",
    "criticality_asset",
    "status_asset"
], errors="ignore")


X = df.drop(columns=[
    "original_priority_score",
    "calculated_priority_score",
    "task_id",
    "asset_id"
], errors="ignore")

y = df["original_priority_score"]


print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


categorical_cols = [
    "department",
    "task_type",
    "section_id",
    "location_code",
    "description",
    "source_defect_id",
    "asset_type",
    "maintenance_type",
    "status",
    "criticality",
    "urgency",
    "operational_impact",
    "priority_category",
    "priority_source",
    "priority_override_reason",
    "required_block_type",
    "can_be_rescheduled"
]



numeric_cols = [
    col for col in X.columns
    if col not in categorical_cols
]




numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_cols
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_cols
    )
])



X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)



model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)


model.fit(
    X_train_processed,
    y_train
)


test_predictions = model.predict(X_test_processed)



mae = mean_absolute_error(
    y_test,
    test_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

r2 = r2_score(
    y_test,
    test_predictions
)


print("\nMODEL PERFORMANCE")

print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)


X_all_processed = preprocessor.transform(X)

all_predictions = model.predict(
    X_all_processed
)


df["predicted_priority_score"] = all_predictions


df.to_csv(  
    file_path,
    index=False
)


print("\nScores saved successfully!")

print("Stored in final_training_data.csv:")
print("- original_priority_score")
print("- calculated_priority_score")
print("- predicted_priority_score")



results = pd.DataFrame({

    "Actual_Priority": y_test.values,

    "Predicted_Priority": test_predictions

})


print("\nActual vs Predicted:")

print(results.head(20))


joblib.dump(
    model,
    "Datasets/New_data/xgboost_priority_model.pkl"
)

joblib.dump(
    preprocessor,
    "Datasets/New_data/priority_preprocessor.pkl"
)

importance = model.feature_importances_

feature_names = (
    preprocessor
    .get_feature_names_out()
)

feature_importance = pd.DataFrame({

    "feature": feature_names,

    "importance": importance

})


feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)


feature_importance.to_csv(
    "Datasets/New_data/feature_importance.csv",
    index=False
)