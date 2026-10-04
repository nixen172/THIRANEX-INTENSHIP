# 🤖 Task 2 — Predictive Modeling Using Machine Learning
**Data Science Internship — Task 2**

---

## 🗂️ Project Structure
```
Task_2/
├── ml_predictive_modeling.py     ← Main script
├── requirements.txt              ← Dependencies
├── README.md                     ← You're here
└── outputs/                      ← Auto-generated after running
    ├── churn_dataset.csv
    ├── model_metrics.csv
    ├── dashboard_1_model_performance.png
    ├── dashboard_2_roc_features.png
    ├── dashboard_3_eda_tree.png
    └── summary_report.txt
```

---

## ⚙️ Setup & Run (VS Code)

### 1. Open the project folder in VS Code
```
File → Open Folder → select Task_2/
```

### 2. Open the integrated terminal
```
Terminal → New Terminal   (or Ctrl + ` )
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the script
```bash
python ml_predictive_modeling.py
```

All output files appear in the **`outputs/`** folder automatically.

---

## 📋 What the Script Does

| Step | Task |
|------|------|
| 1 | Generates a realistic Customer Churn dataset (1000 rows, 9 features) |
| 2 | Preprocesses data — encodes categoricals, splits train/test, scales features |
| 3 | Trains 3 ML models — Logistic Regression, Decision Tree, Random Forest |
| 4 | Evaluates — Accuracy, Precision, Recall, F1, 5-fold Cross-Validation |
| 5 | Generates 3 dashboards with 12+ visualizations |
| 6 | Exports a full summary report |

---

## 📈 Dashboards Generated

### Dashboard 1 — Model Performance Overview
- KPI Cards (Accuracy + F1 per model)
- Grouped bar — Accuracy · Precision · Recall · F1 comparison
- Cross-validation scores with error bars
- Confusion Matrix for all 3 models

### Dashboard 2 — ROC & Feature Importance
- ROC Curves with AUC for all models
- Precision-Recall Curves with AUC
- Random Forest feature importance (horizontal bar)
- Decision Tree feature importance (horizontal bar)

### Dashboard 3 — EDA & Decision Tree
- Churn distribution pie chart
- Tenure vs Churn histogram
- Monthly Charge violin plot (split by churn)
- Support Calls vs Churn Rate bar chart
- Feature Correlation Heatmap
- Decision Tree structure visualization (depth=3)

---

## 🧠 ML Concepts Covered

| Concept | Implementation |
|---------|---------------|
| Train/Test Split | 80/20 with stratification |
| Feature Scaling | `StandardScaler` for Logistic Regression |
| Label Encoding | `LabelEncoder` for categorical features |
| Cross Validation | `StratifiedKFold` (5 folds) |
| Evaluation Metrics | Accuracy, Precision, Recall, F1, AUC-ROC |
| Overfitting Check | CV score vs test score comparison |
| Feature Importance | Built-in (RF & DT) importances |

---

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data handling |
| `numpy` | Numerical ops |
| `matplotlib` | Core plots |
| `seaborn` | Heatmaps, styling |
| `scikit-learn` | All ML models, preprocessing, metrics |
