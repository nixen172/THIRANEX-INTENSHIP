# 📊 Task 3 — Exploratory Data Analysis (EDA) Project
**Data Science Internship — Task 3**

---

## 🗂️ Project Structure
```
Task_3/
├── eda_project.py               ← Main EDA script
├── requirements.txt             ← Dependencies
├── README.md                    ← You're here
└── outputs/                     ← Auto-generated after running
    ├── ecommerce_raw.csv
    ├── ecommerce_cleaned.csv
    ├── statistical_summary.csv
    ├── dashboard_1_overview.png
    ├── dashboard_2_distributions.png
    ├── dashboard_3_customers.png
    ├── dashboard_4_statistics.png
    └── eda_report.txt
```

---

## ⚙️ Setup & Run (VS Code)

### 1. Open folder in VS Code
```
File → Open Folder → select Task_3/
```

### 2. Open terminal
```
Terminal → New Terminal   (Ctrl + `)
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the script
```bash
python eda_project.py
```

All output files appear inside **`outputs/`** automatically.

---

## 📋 What the Script Does

| Step | Task |
|------|------|
| 1 | Generates a realistic E-Commerce dataset (1500 orders, 14 features) |
| 2 | Inspects & cleans — missing values, feature engineering |
| 3 | Statistical summaries — mean, std, skewness, kurtosis, group analysis |
| 4 | Builds 4 dashboards with 20+ visualizations |
| 5 | Exports structured findings & recommendations |

---

## 📈 Dashboards Generated

### Dashboard 1 — Business Overview
- KPI summary cards (Revenue, Orders, AOV, Returns, Rating, Delivery)
- Monthly revenue trend line chart
- Revenue by category (horizontal bar)
- Channel distribution (pie)
- Quarterly revenue vs estimated profit
- Region × Category revenue heatmap
- Orders by weekday

### Dashboard 2 — Distribution & Correlation
- Revenue distribution with KDE overlay
- Age distribution by gender
- Full feature correlation heatmap
- Price boxplots by category
- Discount % vs Revenue scatter (colored by rating)
- Return rate by customer rating

### Dashboard 3 — Customer & Behavioural Insights
- Orders heatmap: Age Group × Category
- Avg revenue by Age Group & Gender
- Delivery time bucket distribution
- Revenue violin plot by sales channel
- Return rate heatmap: Region × Channel
- Auto-generated Key Insights panel

### Dashboard 4 — Statistical Tests & Advanced Patterns
- Q-Q plot (normality check for revenue)
- Price vs Revenue scatter (colored by quantity)
- Weekly revenue + 4-week rolling average
- Discount effectiveness by category (line)
- Delivery days vs customer rating (correlation)
- Cumulative revenue Pareto chart

---

## 🧠 EDA Concepts Covered

| Concept | Technique Used |
|---------|---------------|
| Univariate analysis | Histograms, KDE, Box plots |
| Bivariate analysis | Scatter plots, Grouped bars, Violin |
| Multivariate analysis | Heatmaps, Bubble scatter, Pair trends |
| Time series | Monthly/weekly line charts, Rolling avg |
| Statistical tests | Skewness, Kurtosis, Q-Q plot, Pearson r |
| Business storytelling | Pareto chart, KPI cards, Insight panel |

---

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, grouping |
| `numpy` | Numerical operations |
| `matplotlib` | Core plotting & layout (GridSpec) |
| `seaborn` | Heatmaps, violin, advanced styling |
| `scipy` | Q-Q plots, KDE, statistical functions |
