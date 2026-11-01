# 🚆 Railway Maintenance Priority Score Prediction

### AI-powered maintenance prioritization using XGBoost

An ML-based prototype designed to predict maintenance priority scores for railway assets by combining maintenance task information, asset characteristics, defect records, and train movement data.

The objective is to support better maintenance prioritization by identifying tasks that may require greater attention based on asset conditions, defect severity, maintenance urgency, and operational impact.

---

## 📌 Problem Statement

Railway infrastructure requires regular maintenance to ensure safety, reliability, and uninterrupted train operations.

Maintenance activities involve multiple factors, including:

- Asset criticality and condition
- Maintenance urgency and overdue tasks
- Defect severity and recurrence
- Operational impact
- Train movement and traffic pressure

Evaluating these factors independently can make maintenance prioritization challenging.

This project explores how machine learning can help generate maintenance priority scores by considering multiple operational and asset-related features together.

## 💡 Proposed Solution

The system uses an **XGBoost Regressor** to predict maintenance priority scores based on processed railway maintenance data.

It combines information from four datasets:

1. **Maintenance Tasks** – Maintenance requirements, urgency, status, and due dates.
2. **Assets** – Asset type, criticality, installation details, and maintenance history.
3. **Defects** – Defect severity, overdue defects, and recurring defects.
4. **Train Movements** – Train frequency, freight movements, and train priority information.

The datasets are processed, merged, and transformed into a consolidated training dataset before model development.

## ⚙️ Project Workflow

The project follows three major stages:

### 1. Data Preparation (`prepare_data.py`)

- Loads and integrates the four datasets.
- Performs data cleaning and preprocessing.
- Extracts defect-related features such as total defects, critical defects, and overdue defects.
- Extracts train movement features such as train count and high-priority train count.
- Calculates additional asset and maintenance-related features.
- Generates a calculated priority score using criticality, urgency, operational impact, and overdue maintenance factors.
- Saves the consolidated dataset for model training.

### 2. Model Training (`model.py`)

- Separates input features and target priority scores.
- Splits the dataset into training and testing sets.
- Handles missing numerical and categorical values using `SimpleImputer`.
- Applies `OneHotEncoder` to categorical features.
- Uses `ColumnTransformer` and Scikit-learn pipelines for preprocessing.
- Trains an XGBoost regression model.
- Evaluates performance using:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- Generates predicted priority scores and saves the trained model and preprocessing pipeline using Joblib.

### 3. Final Priority Score Generation (`final_score.py`)

Combines the calculated priority score and ML-predicted priority score to generate a final score.

The implemented weighted formula is:

**Final Priority Score = (0.6 × Calculated Priority Score) + (0.4 × Predicted Priority Score)**

This produces a combined score that considers both the rule-based calculation and the machine learning prediction.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and integration |
| NumPy | Numerical computations |
| Scikit-learn | Preprocessing and evaluation |
| XGBoost | Priority score prediction |
| Joblib | Model and preprocessor serialization |

## 📂 Project Structure

```text
Railway-priority-score-prediction/
│
├── Datasets/
│   └── New_data/
│       ├── assets.csv
│       ├── maintenance_tasks.csv
│       ├── defects.csv
│       ├── train_movements.csv
│       └── final_training_data.csv
│
├── prepare_data.py
├── model.py
├── final_score.py
│
├── priority_preprocessor.pkl
├── xgboost_priority_model.pkl
│
├── requirements.txt
└── README.md
```

## 📊 Model Evaluation

The model is evaluated using the following regression metrics:

- **MAE:** Measures the average absolute difference between actual and predicted priority scores.
- **RMSE:** Measures prediction error while penalizing larger errors.
- **R² Score:** Measures how much variation in the target priority scores is explained by the model.

Actual evaluation results are generated when the model is trained.

## 🎯 Key Learning Outcomes

Through this project, I explored:

- Working with multiple related datasets.
- Feature engineering and data integration.
- Numerical and categorical data preprocessing.
- Building Scikit-learn preprocessing pipelines.
- Implementing XGBoost regression.
- Evaluating regression models.
- Saving and reusing trained ML models.
- Combining rule-based scoring with machine learning predictions.

## ⚠️ Limitations

- This is an experimental prototype and not a deployed railway maintenance management system.
- Predictions depend on the quality and representativeness of the available datasets.
- The generated priority scores are intended for demonstration and analysis, not direct operational decision-making.
- Real-world deployment would require validation using authentic railway operational data, domain-expert review, and integration with existing railway systems.

## 🔮 Future Improvements

- Integrate authentic railway maintenance and operational datasets.
- Improve model validation and hyperparameter tuning.
- Investigate feature importance and model explainability using SHAP.
- Develop a dashboard to visualize maintenance priorities.
- Explore maintenance scheduling and resource allocation optimization.
- Integrate risk prediction to support proactive maintenance planning.

---

## 👨‍💻 Author

**Manas Vinay Kumar**

B.Tech – Computer Science and Engineering

GitHub: [Manas-016](https://github.com/Manas-016)

---

*Developed as an experimental machine learning project exploring intelligent maintenance prioritization for railway infrastructure.*
