"""
=============================================================
  DATA CLEANING & VISUALIZATION PROJECT
  Data Science Internship - Task 1
=============================================================
  Libraries: Pandas, NumPy, Matplotlib, Seaborn
  Run: python data_cleaning_visualization.py
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  STEP 0: SETUP
# ─────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  DATA CLEANING & VISUALIZATION PROJECT")
print("  Data Science Internship — Task 1")
print("=" * 60)


# ─────────────────────────────────────────────
#  STEP 1: GENERATE SYNTHETIC DATASET
#  (Simulates a real-world messy sales dataset)
# ─────────────────────────────────────────────
print("\n📦 STEP 1: Generating raw dataset...\n")

np.random.seed(42)
n = 500

regions      = ["North", "South", "East", "West", "Central"]
products     = ["Laptop", "Phone", "Tablet", "Monitor", "Headphones"]
categories   = {"Laptop": "Electronics", "Phone": "Electronics",
                "Tablet": "Electronics", "Monitor": "Peripherals",
                "Headphones": "Peripherals"}

raw_data = {
    "Order_ID":     [f"ORD-{1000+i}" for i in range(n)],
    "Product":      np.random.choice(products, n),
    "Region":       np.random.choice(regions, n),
    "Sales":        np.random.normal(500, 150, n).round(2),
    "Quantity":     np.random.randint(1, 20, n).astype(float),
    "Discount":     np.random.uniform(0, 0.5, n).round(2),
    "Profit":       np.random.normal(120, 60, n).round(2),
    "Customer_Age": np.random.randint(18, 70, n).astype(float),
    "Rating":       np.random.choice([1,2,3,4,5], n, p=[0.05,0.1,0.2,0.35,0.3]),
    "Order_Date":   pd.date_range("2023-01-01", periods=n, freq="D") +
                    pd.to_timedelta(np.random.randint(0, 365, n), unit="D"),
}

df = pd.DataFrame(raw_data)
df["Category"] = df["Product"].map(categories)

# ── Introduce deliberate messiness ──
df.loc[np.random.choice(n, 50, replace=False), "Sales"]        = np.nan
df.loc[np.random.choice(n, 30, replace=False), "Profit"]       = np.nan
df.loc[np.random.choice(n, 20, replace=False), "Customer_Age"] = np.nan
df.loc[np.random.choice(n, 15, replace=False), "Quantity"]     = np.nan

# Outliers
df.loc[np.random.choice(n, 10, replace=False), "Sales"]   = np.random.uniform(2000, 5000, 10)
df.loc[np.random.choice(n, 5,  replace=False), "Profit"]  = np.random.uniform(-500, -300, 5)

# Duplicates
dup_rows = df.sample(20, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

# String inconsistencies
df.loc[np.random.choice(len(df), 10, replace=False), "Region"] = \
    np.random.choice(["north", "SOUTH", " East "], 10)

df.to_csv(f"{OUTPUT_DIR}/raw_dataset.csv", index=False)
print(f"  ✅ Raw dataset created: {len(df)} rows × {len(df.columns)} columns")
print(f"  📁 Saved → {OUTPUT_DIR}/raw_dataset.csv")


# ─────────────────────────────────────────────
#  STEP 2: DATA INSPECTION
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("📊 STEP 2: Data Inspection")
print("─" * 60)

print(f"\n  Shape       : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Duplicates  : {df.duplicated().sum()}")
print(f"\n  Missing Values:")
missing = df.isnull().sum()
missing = missing[missing > 0]
for col, count in missing.items():
    pct = count / len(df) * 100
    print(f"    {col:<15} → {count:>3} missing  ({pct:.1f}%)")

print(f"\n  Data Types:")
print(df.dtypes.to_string(header=False))


# ─────────────────────────────────────────────
#  STEP 3: DATA CLEANING
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("🧹 STEP 3: Data Cleaning")
print("─" * 60)

df_clean = df.copy()

# 3a. Remove duplicates
before = len(df_clean)
df_clean.drop_duplicates(inplace=True)
print(f"\n  [1] Duplicates removed  : {before - len(df_clean)} rows")

# 3b. Standardize text
df_clean["Region"] = df_clean["Region"].str.strip().str.title()
print(f"  [2] Region values fixed : {df_clean['Region'].unique()}")

# 3c. Handle missing values
df_clean["Sales"].fillna(df_clean["Sales"].median(), inplace=True)
df_clean["Profit"].fillna(df_clean["Profit"].median(), inplace=True)
df_clean["Quantity"].fillna(df_clean["Quantity"].mode()[0], inplace=True)
df_clean["Customer_Age"].fillna(df_clean["Customer_Age"].median(), inplace=True)
print(f"  [3] Missing values filled with median/mode")

# 3d. Handle outliers using IQR method
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower, upper)
    return df, outliers

df_clean, sales_out  = remove_outliers_iqr(df_clean, "Sales")
df_clean, profit_out = remove_outliers_iqr(df_clean, "Profit")
print(f"  [4] Outliers capped (IQR): Sales={sales_out}, Profit={profit_out}")

# 3e. Add engineered features
df_clean["Revenue"]         = df_clean["Sales"] * df_clean["Quantity"]
df_clean["Profit_Margin"]   = (df_clean["Profit"] / df_clean["Revenue"] * 100).round(2)
df_clean["Month"]           = df_clean["Order_Date"].dt.month_name()
df_clean["Quarter"]         = "Q" + df_clean["Order_Date"].dt.quarter.astype(str)
df_clean["Age_Group"]       = pd.cut(df_clean["Customer_Age"],
                                     bins=[18,30,45,60,70],
                                     labels=["18-30","31-45","46-60","61-70"])

df_clean.to_csv(f"{OUTPUT_DIR}/cleaned_dataset.csv", index=False)
print(f"\n  ✅ Cleaned dataset: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
print(f"  📁 Saved → {OUTPUT_DIR}/cleaned_dataset.csv")


# ─────────────────────────────────────────────
#  STEP 4: EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔍 STEP 4: Exploratory Data Analysis")
print("─" * 60)

print("\n  Descriptive Statistics (numeric):")
print(df_clean[["Sales","Profit","Quantity","Revenue","Profit_Margin"]].describe().round(2).to_string())

print("\n  Revenue by Product:")
rev_product = df_clean.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
for prod, rev in rev_product.items():
    print(f"    {prod:<15} ₹{rev:>12,.0f}")

print("\n  Average Rating by Region:")
rating_region = df_clean.groupby("Region")["Rating"].mean().round(2)
for reg, rat in rating_region.items():
    stars = "★" * round(rat)
    print(f"    {reg:<10} {rat}  {stars}")


# ─────────────────────────────────────────────
#  STEP 5: VISUALIZATIONS — DASHBOARD 1
#  Overview Dashboard
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("📈 STEP 5: Generating Visualizations...")
print("─" * 60)

plt.rcParams.update({
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#1A1D27",
    "axes.edgecolor":   "#2E3347",
    "axes.labelcolor":  "#C8CFDF",
    "xtick.color":      "#8892A4",
    "ytick.color":      "#8892A4",
    "text.color":       "#E2E8F4",
    "grid.color":       "#2E3347",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "DejaVu Sans",
})

ACCENT     = "#7C6AF5"
ACCENT2    = "#F5856A"
ACCENT3    = "#52D9A3"
ACCENT4    = "#F5C842"
PALETTE    = [ACCENT, ACCENT2, ACCENT3, ACCENT4, "#A78BFA"]

# ── DASHBOARD 1: Overview ──────────────────────────
fig = plt.figure(figsize=(20, 14), facecolor="#0F1117")
fig.suptitle("📊  SALES PERFORMANCE DASHBOARD", fontsize=22, fontweight="bold",
             color="#E2E8F4", y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# KPI Cards (top row, col 0-2)
kpis = [
    ("Total Revenue",    f"₹{df_clean['Revenue'].sum()/1e6:.2f}M",  ACCENT),
    ("Total Profit",     f"₹{df_clean['Profit'].sum()/1e3:.1f}K",   ACCENT3),
    ("Avg Order Value",  f"₹{df_clean['Sales'].mean():.0f}",        ACCENT2),
    ("Total Orders",     f"{len(df_clean):,}",                      ACCENT4),
    ("Avg Rating",       f"{df_clean['Rating'].mean():.2f} ★",      "#A78BFA"),
    ("Avg Discount",     f"{df_clean['Discount'].mean()*100:.1f}%", "#F28CB1"),
]

# KPI row using axes
for i, (label, value, color) in enumerate(kpis):
    col = i % 3
    row = i // 3
    if row == 0:
        ax_kpi = fig.add_subplot(gs[0, col])
        ax_kpi.set_xlim(0, 1); ax_kpi.set_ylim(0, 1)
        ax_kpi.axis("off")
        ax_kpi.add_patch(plt.Rectangle((0,0), 1, 1, color="#1E2235", zorder=0,
                                        transform=ax_kpi.transData))
        ax_kpi.text(0.5, 0.62, value,  ha="center", va="center", fontsize=22,
                    fontweight="bold", color=color, transform=ax_kpi.transAxes)
        ax_kpi.text(0.5, 0.28, label,  ha="center", va="center", fontsize=11,
                    color="#8892A4", transform=ax_kpi.transAxes)
        ax_kpi.patch.set_facecolor("#1E2235")

# 1. Revenue by Product (bar)
ax1 = fig.add_subplot(gs[1, 0])
rev = df_clean.groupby("Product")["Revenue"].sum().sort_values()
bars = ax1.barh(rev.index, rev.values / 1000, color=PALETTE[:len(rev)], edgecolor="none")
for bar, val in zip(bars, rev.values):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f"₹{val/1e3:.0f}K", va="center", fontsize=9, color="#C8CFDF")
ax1.set_title("Revenue by Product", fontweight="bold", color="#E2E8F4", pad=8)
ax1.set_xlabel("Revenue (₹ Thousands)")
ax1.grid(axis="x", alpha=0.3)

# 2. Sales Distribution (histogram)
ax2 = fig.add_subplot(gs[1, 1])
ax2.hist(df_clean["Sales"], bins=30, color=ACCENT, edgecolor="#0F1117", alpha=0.85)
ax2.axvline(df_clean["Sales"].mean(), color=ACCENT2, linestyle="--", linewidth=1.5,
            label=f'Mean: ₹{df_clean["Sales"].mean():.0f}')
ax2.axvline(df_clean["Sales"].median(), color=ACCENT3, linestyle="--", linewidth=1.5,
            label=f'Median: ₹{df_clean["Sales"].median():.0f}')
ax2.legend(fontsize=8, facecolor="#1A1D27", edgecolor="#2E3347", labelcolor="#C8CFDF")
ax2.set_title("Sales Distribution", fontweight="bold", color="#E2E8F4", pad=8)
ax2.set_xlabel("Sales (₹)")

# 3. Region Pie
ax3 = fig.add_subplot(gs[1, 2])
region_rev = df_clean.groupby("Region")["Revenue"].sum()
wedges, texts, autotexts = ax3.pie(region_rev, labels=region_rev.index,
                                    autopct="%1.1f%%", colors=PALETTE,
                                    startangle=140, pctdistance=0.75,
                                    wedgeprops=dict(edgecolor="#0F1117", linewidth=1.5))
for t in texts:     t.set_color("#C8CFDF"); t.set_fontsize(9)
for t in autotexts: t.set_color("#0F1117"); t.set_fontsize(8); t.set_fontweight("bold")
ax3.set_title("Revenue by Region", fontweight="bold", color="#E2E8F4", pad=8)

# 4. Monthly Revenue Trend (line)
ax4 = fig.add_subplot(gs[2, 0:2])
month_order = ["January","February","March","April","May","June","July",
               "August","September","October","November","December"]
monthly = df_clean.groupby("Month")["Revenue"].sum().reindex(
    [m for m in month_order if m in df_clean["Month"].unique()])
ax4.plot(range(len(monthly)), monthly.values / 1000,
         color=ACCENT, linewidth=2.5, marker="o", markersize=6,
         markerfacecolor=ACCENT2, markeredgecolor="#0F1117", markeredgewidth=1.5)
ax4.fill_between(range(len(monthly)), monthly.values / 1000, alpha=0.15, color=ACCENT)
ax4.set_xticks(range(len(monthly)))
ax4.set_xticklabels([m[:3] for m in monthly.index], rotation=30, ha="right")
ax4.set_title("Monthly Revenue Trend", fontweight="bold", color="#E2E8F4", pad=8)
ax4.set_ylabel("Revenue (₹ Thousands)")
ax4.grid(axis="y", alpha=0.3)

# 5. Rating Distribution
ax5 = fig.add_subplot(gs[2, 2])
rating_counts = df_clean["Rating"].value_counts().sort_index()
bars5 = ax5.bar(rating_counts.index, rating_counts.values,
                color=[PALETTE[i % len(PALETTE)] for i in range(len(rating_counts))],
                edgecolor="#0F1117", width=0.6)
ax5.set_title("Customer Rating Distribution", fontweight="bold", color="#E2E8F4", pad=8)
ax5.set_xlabel("Rating (★)")
ax5.set_ylabel("Count")
for bar in bars5:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(int(bar.get_height())), ha="center", fontsize=9, color="#C8CFDF")

plt.savefig(f"{OUTPUT_DIR}/dashboard_1_overview.png", dpi=150, bbox_inches="tight",
            facecolor="#0F1117")
plt.close()
print(f"  ✅ Dashboard 1 → {OUTPUT_DIR}/dashboard_1_overview.png")


# ── DASHBOARD 2: Deep-Dive Analysis ────────────────
fig2, axes = plt.subplots(2, 3, figsize=(20, 11), facecolor="#0F1117")
fig2.suptitle("🔍  DEEP-DIVE ANALYSIS DASHBOARD", fontsize=20, fontweight="bold",
              color="#E2E8F4", y=1.01)

# 1. Correlation Heatmap
ax = axes[0][0]
numeric_cols = ["Sales","Profit","Quantity","Discount","Revenue","Customer_Age","Rating"]
corr = df_clean[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr, ax=ax, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            annot=True, fmt=".2f", linewidths=0.5, linecolor="#0F1117",
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.7})
ax.set_title("Correlation Heatmap", fontweight="bold", color="#E2E8F4", pad=8)
ax.tick_params(colors="#8892A4")

# 2. Profit by Category & Product
ax = axes[0][1]
cat_profit = df_clean.groupby(["Category","Product"])["Profit"].mean().reset_index()
colors_map = {"Electronics": ACCENT, "Peripherals": ACCENT2}
bar_colors = [colors_map[c] for c in cat_profit["Category"]]
bars = ax.bar(cat_profit["Product"], cat_profit["Profit"],
              color=bar_colors, edgecolor="#0F1117", width=0.6)
ax.axhline(df_clean["Profit"].mean(), color=ACCENT3, linestyle="--",
           linewidth=1.5, label="Overall Mean")
ax.legend(fontsize=8, facecolor="#1A1D27", edgecolor="#2E3347", labelcolor="#C8CFDF")
ax.set_title("Avg Profit by Product", fontweight="bold", color="#E2E8F4", pad=8)
ax.set_ylabel("Avg Profit (₹)")
ax.tick_params(axis="x", rotation=15)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=ACCENT, label="Electronics"),
                   Patch(facecolor=ACCENT2, label="Peripherals")]
ax.legend(handles=legend_elements, fontsize=8, facecolor="#1A1D27",
          edgecolor="#2E3347", labelcolor="#C8CFDF")

# 3. Sales vs Profit Scatter
ax = axes[0][2]
product_list = df_clean["Product"].unique()
for i, prod in enumerate(product_list):
    subset = df_clean[df_clean["Product"] == prod]
    ax.scatter(subset["Sales"], subset["Profit"],
               color=PALETTE[i % len(PALETTE)], alpha=0.5, s=18, label=prod,
               edgecolors="none")
ax.axhline(0, color=ACCENT2, linewidth=1, linestyle="--", alpha=0.6)
ax.set_title("Sales vs Profit", fontweight="bold", color="#E2E8F4", pad=8)
ax.set_xlabel("Sales (₹)"); ax.set_ylabel("Profit (₹)")
ax.legend(fontsize=7, facecolor="#1A1D27", edgecolor="#2E3347",
          labelcolor="#C8CFDF", markerscale=1.5)

# 4. Quarterly Performance
ax = axes[1][0]
quarters = ["Q1","Q2","Q3","Q4"]
q_data = df_clean[df_clean["Quarter"].isin(quarters)]
q_rev = q_data.groupby("Quarter")["Revenue"].sum() / 1000
q_prof = q_data.groupby("Quarter")["Profit"].sum() / 1000
x = np.arange(len(q_rev))
w = 0.35
ax.bar(x - w/2, q_rev, w, label="Revenue", color=ACCENT, edgecolor="#0F1117")
ax.bar(x + w/2, q_prof, w, label="Profit",  color=ACCENT3, edgecolor="#0F1117")
ax.set_xticks(x); ax.set_xticklabels(q_rev.index)
ax.set_title("Quarterly Revenue vs Profit", fontweight="bold", color="#E2E8F4", pad=8)
ax.set_ylabel("Amount (₹ Thousands)")
ax.legend(fontsize=8, facecolor="#1A1D27", edgecolor="#2E3347", labelcolor="#C8CFDF")
ax.grid(axis="y", alpha=0.3)

# 5. Customer Age Group × Rating Heatmap
ax = axes[1][1]
age_rating = df_clean.groupby(["Age_Group","Rating"])["Order_ID"].count().unstack(fill_value=0)
sns.heatmap(age_rating, ax=ax, cmap="YlOrRd", annot=True, fmt="d",
            linewidths=0.5, linecolor="#0F1117",
            cbar_kws={"shrink": 0.7})
ax.set_title("Age Group vs Rating", fontweight="bold", color="#E2E8F4", pad=8)
ax.tick_params(colors="#8892A4")
ax.set_xlabel("Rating (★)"); ax.set_ylabel("Age Group")

# 6. Discount vs Profit Margin
ax = axes[1][2]
sc = ax.scatter(df_clean["Discount"] * 100, df_clean["Profit_Margin"],
                c=df_clean["Sales"], cmap="plasma", alpha=0.4, s=20,
                edgecolors="none")
cb = plt.colorbar(sc, ax=ax)
cb.set_label("Sales (₹)", color="#C8CFDF", fontsize=8)
cb.ax.yaxis.set_tick_params(color="#8892A4", labelcolor="#8892A4")
# Trend line
z = np.polyfit(df_clean["Discount"], df_clean["Profit_Margin"].fillna(0), 1)
p = np.poly1d(z)
xline = np.linspace(df_clean["Discount"].min(), df_clean["Discount"].max(), 100)
ax.plot(xline * 100, p(xline), color=ACCENT2, linewidth=2, linestyle="--")
ax.set_title("Discount % vs Profit Margin", fontweight="bold", color="#E2E8F4", pad=8)
ax.set_xlabel("Discount (%)"); ax.set_ylabel("Profit Margin (%)")
ax.grid(alpha=0.3)

for ax_row in axes:
    for ax_ in ax_row:
        ax_.set_facecolor("#1A1D27")
        ax_.spines["bottom"].set_color("#2E3347")
        ax_.spines["left"].set_color("#2E3347")
        ax_.spines["top"].set_visible(False)
        ax_.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dashboard_2_deepdive.png", dpi=150, bbox_inches="tight",
            facecolor="#0F1117")
plt.close()
print(f"  ✅ Dashboard 2 → {OUTPUT_DIR}/dashboard_2_deepdive.png")


# ── DASHBOARD 3: Before vs After Cleaning ──────────
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0F1117")
fig3.suptitle("🧹  BEFORE vs AFTER DATA CLEANING", fontsize=18, fontweight="bold",
              color="#E2E8F4", y=1.02)

for i, (data, label, color) in enumerate([
    (df["Sales"].dropna(),       "RAW Data (with outliers)", ACCENT2),
    (df_clean["Sales"],          "CLEANED Data",             ACCENT3)
]):
    ax = axes3[i]
    ax.set_facecolor("#1A1D27")
    ax.hist(data, bins=40, color=color, edgecolor="#0F1117", alpha=0.85)
    ax.axvline(data.mean(),   color="#FFFFFF", linestyle="--", linewidth=1.5,
               label=f"Mean: ₹{data.mean():.0f}")
    ax.axvline(data.median(), color=ACCENT4,  linestyle="-",  linewidth=1.5,
               label=f"Median: ₹{data.median():.0f}")
    ax.set_title(label, fontweight="bold", color="#E2E8F4", pad=10, fontsize=13)
    ax.set_xlabel("Sales (₹)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9, facecolor="#1A1D27", edgecolor="#2E3347", labelcolor="#C8CFDF")
    ax.grid(axis="y", alpha=0.3)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#2E3347")
    ax.spines["left"].set_color("#2E3347")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dashboard_3_before_after.png", dpi=150, bbox_inches="tight",
            facecolor="#0F1117")
plt.close()
print(f"  ✅ Dashboard 3 → {OUTPUT_DIR}/dashboard_3_before_after.png")


# ─────────────────────────────────────────────
#  STEP 6: SUMMARY REPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("📋 SUMMARY REPORT")
print("=" * 60)

report = f"""
╔══════════════════════════════════════════════════════════╗
║         DATA CLEANING & VISUALIZATION REPORT             ║
╠══════════════════════════════════════════════════════════╣
║  RAW DATASET                                             ║
║    Rows       : {len(df):<6}  Columns : {len(df.columns):<6}                   ║
║    Duplicates : {df.duplicated().sum():<6}  Missing : {df.isnull().sum().sum():<6}                   ║
╠══════════════════════════════════════════════════════════╣
║  CLEANED DATASET                                         ║
║    Rows       : {len(df_clean):<6}  Columns : {len(df_clean.columns):<6}                   ║
║    Missing    : {df_clean.isnull().sum().sum():<6}  Outliers: Capped (IQR)         ║
╠══════════════════════════════════════════════════════════╣
║  KEY INSIGHTS                                            ║
║    Total Revenue  : ₹{df_clean["Revenue"].sum():>12,.0f}                  ║
║    Total Profit   : ₹{df_clean["Profit"].sum():>12,.0f}                  ║
║    Avg Sale Value : ₹{df_clean["Sales"].mean():>12,.2f}                  ║
║    Avg Rating     :  {df_clean["Rating"].mean():>5.2f} / 5.00                      ║
║    Best Product   :  {rev_product.index[0]:<20}              ║
║    Best Region    :  {df_clean.groupby("Region")["Revenue"].sum().idxmax():<20}              ║
╠══════════════════════════════════════════════════════════╣
║  OUTPUT FILES                                            ║
║    outputs/raw_dataset.csv                               ║
║    outputs/cleaned_dataset.csv                           ║
║    outputs/dashboard_1_overview.png                      ║
║    outputs/dashboard_2_deepdive.png                      ║
║    outputs/dashboard_3_before_after.png                  ║
╚══════════════════════════════════════════════════════════╝
"""
print(report)

with open(f"{OUTPUT_DIR}/summary_report.txt", "w") as f:
    f.write(report)

print(f"  📁 Report saved → {OUTPUT_DIR}/summary_report.txt")
print("\n✅  All done! Check the 'outputs/' folder.\n")
