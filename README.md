# README - Predicting Customer Responses to Term Deposit Campaigns

## Project Overview
This project leverages **machine learning** to predict customer responses to **bank term deposit campaigns**. 
By analyzing historical marketing data, we aim to help banks **improve customer targeting, optimize outreach strategies, and increase conversion rates**.

## Why This Project?
Traditional marketing campaigns often **waste resources on uninterested customers**, leading to **high costs and low efficiency**. 
By implementing **data-driven decision-making**, banks can **identify high-potential customers** and refine their marketing strategies.

## Files in This Repository
- **509_Project_Report.pdf** – Full project documentation, including methodology, results, and insights.
- **Code.ipynb** – Jupyter Notebook containing the complete machine learning workflow.
- **bank-full.csv** – Dataset used for training and evaluation.
- **README.txt** – This file, explaining the project structure and how to reproduce results.

## Dataset Overview
- **Source:** Bank Marketing Dataset
- **Size:** 45,211 rows, 17 columns
- **Key Features:**
  - **Demographic:** Age, job, marital status, education
  - **Financial:** Account balance, housing loan, personal loan
  - **Marketing:** Contact method, call duration, past interactions
  - **Target Variable:** `subscribed` (Yes/No)

## Machine Learning Pipeline
1. **Data Preprocessing**
   - Categorical features: **One-Hot Encoding (job, marital)**, **Ordinal Encoding (education)**
   - Numerical features: **Standardized with `StandardScaler()`**
   - Dropped non-informative features (**day, month**) and those causing data leakage (**call duration**).

2. **Model Selection & Hyperparameter Tuning**
   - Evaluated **K-Nearest Neighbors (KNN), Logistic Regression, and Random Forest**.
   - Used **GridSearchCV** for hyperparameter tuning.
   - Selected **Random Forest** as the best-performing model.

3. **Feature Importance Analysis**
   - Applied **Permutation Importance** to identify top predictors.
   - Key features: **Recent contact (`pdays`), previous interactions (`previous`), and customer engagement metrics.**

## How to Reproduce Results
To **reproduce the model and analysis**, follow these steps:

### 1. Install Dependencies
Ensure Python and required libraries are installed:
```
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Load the Dataset
Place `bank-full.csv` in the working directory and run:
```
import pandas as pd
df = pd.read_csv("bank-full.csv", delimiter=";")
```

### 3. Run the Jupyter Notebook
Open `Code.ipynb` and execute the cells sequentially to:
- **Preprocess the dataset**
- **Train and evaluate models**
- **Perform feature importance analysis**
- **Generate marketing insights**

### 4. Train & Evaluate the Model
To run the best model (**Random Forest**), execute:
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

rf_pipeline = make_pipeline(preprocessor, RandomForestClassifier(random_state=42, class_weight={"no":1, "yes":10}))

param_grid = {
    "randomforestclassifier__n_estimators": [100, 200, 300, 400, 500],
    "randomforestclassifier__max_depth": [3, None]
}

grid_search = GridSearchCV(rf_pipeline, param_grid, scoring="accuracy", cv=5)
grid_search.fit(train_df.drop(columns=["subscribed"]), train_df["subscribed"])
best_rf_model = grid_search.best_estimator_
```

### 5. Evaluate Feature Importance
To determine the most influential features, run:
```
feature_names = preprocessor.get_feature_names_out()
feature_importances = best_rf_model.named_steps["randomforestclassifier"].feature_importances_

import pandas as pd
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

print(feature_importance_df.head(10))  # Display top 10 most important features
```

## Key Findings & Business Impact
- **Timely follow-ups** significantly improve subscription rates.
- **Repeated interactions** enhance customer trust and engagement.
- **Behavioral data** (past engagement) is more predictive than financial indicators.
- **Targeting strategies should integrate behavioral and financial metrics** for better precision.

## Future Considerations
- Experiment with **Neural Networks (MLP, RNNs)** for deeper insights.
- Implement **cost-sensitive learning** to minimize False Negatives.
- Use **explainable AI methods (SHAP, LIME)** to enhance model transparency.


