# Adult Income Prediction Pipeline

## 📌 Project Overview
This project implements a complete end-to-end machine learning pipeline to predict whether an individual's annual income exceeds $50,000 based on census data (UCI Adult Dataset). The solution covers data loading, Exploratory Data Analysis (EDA), preprocessing, model training, hyperparameter tuning, and advanced interpretability using SHAP values.

## 🎯 Goals
- Predict income class (`<=50K` vs `>50K`) using census data.
- Handle missing values, categorical encoding, and feature scaling.
- Address class imbalance using **SMOTE**.
- Compare multiple models: **Logistic Regression**, **Random Forest**, and **XGBoost**.
- Perform hyperparameter tuning with **GridSearchCV**.
- Interpret model decisions using **SHAP** (SHapley Additive exPlanations).

## 📂 File Structure
```
adult-income-prediction/
│   ├── adult_income_data.csv   # Preprocessed dataset (snake_case headers)
│   ├── notebook.ipynb          # Main Jupyter/Colab notebook
│   ├── README.md               # This file
│   └── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn ydata-profiling shap
```

### 2. Run the Notebook

**Option A: Google Colab (Recommended)**
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `notebook.ipynb` and `adult_income_data.csv`.
3. Run all cells (`Runtime > Run All`).

**Option B: Local Jupyter**
1. Ensure dependencies are installed.
2. Launch Jupyter: `jupyter notebook`
3. Open `notebook.ipynb` and run all cells.

## 🔧 Pipeline Steps
- **Data Loading**: Reads CSV, handles `?` as missing values.
- **EDA**: Generates interactive reports with `ydata-profiling`.
- **Preprocessing**:
  - Numeric: Median imputation + Standard Scaling.
  - Categorical: Mode imputation + One-Hot Encoding.
- **Imbalance Handling**: Applies **SMOTE** to oversample the minority class.
- **Model Training**: Trains **Logistic Regression**, **Random Forest**, and **XGBoost**.
- **Evaluation**:
  - Metrics: ROC-AUC, Precision, Recall, F1-Score.
  - Cross-Validation: 5-fold CV scores.
  - Learning Curves: Bias/Variance diagnosis.
- **Tuning**: `GridSearchCV` for the best model.
- **Interpretability**: **SHAP** summary plots and feature importance.

## 📊 Dataset Details
- **Source**: UCI Machine Learning Repository (Adult/Census Income).
- **Features**: 14 attributes (Age, Workclass, Education, Occupation, etc.).
- **Target**: `income` (Binary: `<=50K` or `>50K`).
- **Preprocessing**: Headers converted to snake_case (e.g., `capital_loss`, `native_country`).

## 📈 Key Outputs
- Interactive EDA HTML report (if saved).
- Model comparison charts (ROC Curves).
- Confusion Matrices.
- Learning Curves.
- SHAP Summary Plots (Bar & Bee Swarm).
- Top 10 Feature Importance rankings.

## 📝 Requirements
- Python `>= 3.8`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `xgboost`, `imbalanced-learn`
- `ydata-profiling`, `shap`

## 📄 License
Educational use only. Dataset licensed under CC BY 4.0.
