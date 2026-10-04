# Task 4 — Real-World Data Project (Health Domain)
## Breast Cancer Diagnosis: End-to-End Data Analysis & Prediction

This project applies a full data science workflow — EDA, visualization, and
predictive modeling — to a real, publicly known clinical dataset, as required
for Internship Task 4.

## 1. Dataset

**Breast Cancer Wisconsin (Diagnostic) Data Set**
- Real-world dataset originally from the UCI Machine Learning Repository,
  bundled with `scikit-learn` (`sklearn.datasets.load_breast_cancer`).
- 569 patient records, 30 numeric features per record describing cell-nuclei
  measurements taken from digitized images of breast mass biopsies (fine
  needle aspirate / FNA).
- Target: **diagnosis** — Malignant or Benign.
- Saved locally as `data/breast_cancer_data.csv` when the project is run, so
  it behaves like any other on-disk dataset.

## 2. Project Structure

```
Task 4/
├── README.md                 <- you are here
├── requirements.txt          <- Python dependencies
├── data/
│   └── breast_cancer_data.csv        <- dataset (generated on first run, also pre-included)
├── notebooks/
│   └── Task4_Breast_Cancer_Analysis.ipynb   <- interactive notebook (run this in VS Code)
├── src/
│   └── analysis.py           <- same pipeline as a plain script
└── outputs/                  <- all charts, metrics, conclusions, trained model
    ├── 01_class_distribution.png
    ├── 02_correlation_heatmap.png
    ├── 03_feature_distributions.png
    ├── 04_boxplots.png
    ├── 05_confusion_matrix_logistic_regression.png
    ├── 05_confusion_matrix_random_forest.png
    ├── 06_roc_curve_comparison.png
    ├── 07_feature_importance.png
    ├── model_results.json
    ├── conclusions.txt
    ├── random_forest_model.joblib
    └── feature_scaler.joblib
```

## 3. How to Run in VS Code

1. Unzip the folder and open `Task 4/` in VS Code (`File → Open Folder...`).
2. Make sure Python 3.9+ is installed, then create/activate a virtual
   environment (recommended):
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **Option A — Run the script** (fastest, regenerates everything in `outputs/`):
   ```bash
   python src/analysis.py
   ```
5. **Option B — Run the notebook** (interactive, recommended for review):
   - Install the VS Code "Python" and "Jupyter" extensions if you don't have them.
   - Open `notebooks/Task4_Breast_Cancer_Analysis.ipynb`.
   - Click **Run All**. Each cell mirrors a step of the pipeline and displays
     its chart inline, while also saving a copy into `outputs/`.

All outputs in this submission were already generated once by running
`src/analysis.py`, so you can review the charts and results immediately
without running anything, and re-run them yourself to confirm reproducibility.

## 4. Methodology

1. **Load & save data** — load the dataset via scikit-learn, write to CSV.
2. **EDA** — check for missing values, examine class balance, plot a
   correlation heatmap, feature distributions by diagnosis, and boxplots to
   spot outliers.
3. **Preprocessing** — stratified 80/20 train/test split, then standardize
   features with `StandardScaler`.
4. **Modeling** — train two classifiers: Logistic Regression (simple,
   interpretable baseline) and Random Forest (captures non-linear patterns,
   exposes feature importance).
5. **Evaluation** — accuracy, precision, recall, F1-score, ROC-AUC, confusion
   matrices, and an ROC curve comparison between the two models.
6. **Feature importance** — identify which measurements most influence the
   Random Forest's predictions.
7. **Conclusions** — written summary in `outputs/conclusions.txt`.

## 5. Key Findings

- The dataset is moderately imbalanced (~63% Benign / 37% Malignant); a
  stratified split kept this ratio consistent between train and test sets.
- "Mean" tumor-size features (radius, perimeter, area, concavity) are
  strongly correlated with each other and with malignancy: larger, more
  irregular cells point toward malignant tumors.
- Both models scored above **95% accuracy** and above **0.99 ROC-AUC** on the
  held-out test set; exact numbers are saved in `outputs/model_results.json`
  and may vary slightly by scikit-learn version.
- The top predictive features were consistently the "worst" and "mean"
  versions of perimeter, area, radius, and concave points.
- **Clinical takeaway:** recall on the Malignant class is the most important
  metric here, since missing a malignant case (a false negative) is far more
  costly than a false alarm — any real deployment should tune the decision
  threshold with this asymmetry in mind, rather than optimizing for raw
  accuracy alone.

Full numeric results and the written conclusion are in `outputs/conclusions.txt`
and `outputs/model_results.json`.

## 6. Notes for Reviewers

- The dataset is genuine, not synthetic — chosen specifically so the project
  demonstrates skills on real clinical data without depending on an external
  download (it ships locally with scikit-learn).
- Both a script and a notebook are provided so the project can be run either
  way depending on the reviewer's preference.
