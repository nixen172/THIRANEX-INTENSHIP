# 📊 Data Cleaning & Visualization Project
**Data Science Internship — Task 1**

---

## 🗂️ Project Structure
```
data_cleaning_project/
├── data_cleaning_visualization.py   ← Main script
├── requirements.txt                 ← Dependencies
├── README.md                        ← You're here
└── outputs/                         ← Auto-generated after running
    ├── raw_dataset.csv
    ├── cleaned_dataset.csv
    ├── dashboard_1_overview.png
    ├── dashboard_2_deepdive.png
    ├── dashboard_3_before_after.png
    └── summary_report.txt
```

---

## ⚙️ Setup & Run (VS Code)

### 1. Open the project folder in VS Code
```
File → Open Folder → select data_cleaning_project/
```

### 2. Open the integrated terminal
```
Terminal → New Terminal   (or Ctrl + `)
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the script
```bash
python data_cleaning_visualization.py
```

All output files are saved inside the **`outputs/`** folder automatically.

---

## 📋 What the Script Does

| Step | Task |
|------|------|
| 1 | Generates a realistic messy sales dataset (500 rows) |
| 2 | Inspects data — shape, dtypes, missing values, duplicates |
| 3 | Cleans data — removes duplicates, fills missing values, caps outliers, standardizes text |
| 4 | Engineers new features — Revenue, Profit Margin, Month, Quarter, Age Group |
| 5 | Creates 3 dashboards with 12+ charts |
| 6 | Exports a summary report |

---

## 📈 Dashboards Generated

### Dashboard 1 — Overview
- KPI Cards (Revenue, Profit, Orders, Rating, Discount)
- Revenue by Product (horizontal bar)
- Sales Distribution (histogram with mean/median)
- Revenue by Region (pie chart)
- Monthly Revenue Trend (line chart)
- Customer Rating Distribution (bar chart)

### Dashboard 2 — Deep-Dive Analysis
- Correlation Heatmap
- Avg Profit by Product & Category
- Sales vs Profit Scatter (colored by product)
- Quarterly Revenue vs Profit (grouped bar)
- Age Group × Rating Heatmap
- Discount % vs Profit Margin (scatter + trend line)

### Dashboard 3 — Before vs After Cleaning
- Side-by-side Sales histograms showing the effect of cleaning

---

## 🧹 Cleaning Techniques Used

- **Duplicate removal** — `drop_duplicates()`
- **Missing values** — filled with median (numeric) / mode (categorical)
- **Outlier handling** — IQR method with clipping
- **Text standardization** — `.str.strip().str.title()`
- **Feature engineering** — Revenue, Margin, Month, Quarter, Age Group

---

## 🛠️ Libraries Used
| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, transformation |
| `numpy` | Numerical operations, outlier detection |
| `matplotlib` | Core plotting engine |
| `seaborn` | Heatmaps, statistical plots |
