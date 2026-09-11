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

df = pd.read_csv("Datasets/final_training_data.csv")
df = df.drop(columns=[
    "department_asset",
    "section_id_asset",
    "location_code_asset",
    "criticality_asset",
    "status_asset"
])
print(df.shape)
print(df.columns.tolist())

X = df.drop(columns=["priority_score", "task_id", "asset_id"])
y = df["priority_score"]

X_train ,X_test , y_train , y_test = train_test_split(X, y, test_size = 0.2 , random_state = 42)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


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

numeric_cols =  [col for col in X.columns if col not in categorical_cols]

numeric_pipeline = Pipeline([
    ("imputer" , SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(X_train_processed.shape)

model = XGBRegressor(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
                     objective="reg:squarederror",random_state=42)

model.fit(X_train_processed,y_train)

predictions = model.predict(X_test_processed)

mae = mean_absolute_error(y_test , predictions)
rmse = np.sqrt(mean_squared_error(y_test , predictions))
r2 = r2_score(y_test , predictions)

print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

results = pd.DataFrame({
    "Actual_Priority": y_test.values,
    "Predicted_Priority": predictions
})

print(results.head(20))

joblib.dump(model, "xgboost_priority_model.pkl")
joblib.dump(preprocessor, "priority_preprocessor.pkl")



