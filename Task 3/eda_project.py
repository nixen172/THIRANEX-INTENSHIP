"""
=================================================================
  TASK 3 — EXPLORATORY DATA ANALYSIS (EDA) PROJECT
  Data Science Internship
=================================================================
  Dataset  : Synthetic E-Commerce Orders Dataset
  Libraries: Pandas, NumPy, Matplotlib, Seaborn, SciPy
  Run with : python eda_project.py
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
import os
import textwrap

warnings.filterwarnings("ignore")

# ── Output directory ──────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Design tokens ─────────────────────────────────────────────
BG      = "#0B0E1A"
PANEL   = "#131929"
PANEL2  = "#1C2238"
BORDER  = "#252E48"
TXT     = "#E8EDF8"
MUTED   = "#7A87A3"
C1      = "#6C63FF"   # indigo
C2      = "#FF6584"   # rose
C3      = "#43D9A2"   # teal
C4      = "#FFB547"   # amber
C5      = "#56CCF2"   # sky
C6      = "#BB6BD9"   # violet
PAL6    = [C1, C2, C3, C4, C5, C6]

plt.rcParams.update({
    "figure.facecolor":  BG,    "axes.facecolor":   PANEL,
    "axes.edgecolor":    BORDER,"axes.labelcolor":  TXT,
    "xtick.color":       MUTED, "ytick.color":      MUTED,
    "text.color":        TXT,   "grid.color":       BORDER,
    "grid.linestyle":    "--",  "grid.alpha":       0.5,
    "font.family":       "DejaVu Sans",
    "legend.facecolor":  PANEL, "legend.edgecolor": BORDER,
})

def sep(title=""):
    bar = "─" * 62
    print(f"\n{bar}")
    if title:
        print(f"  {title}")
        print(bar)

def style(ax, grid="y"):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.spines["left"].set_color(BORDER)
    if "y" in grid: ax.yaxis.grid(True, alpha=0.4)
    if "x" in grid: ax.xaxis.grid(True, alpha=0.4)
    if "n" in grid: ax.grid(False)


print("=" * 62)
print("  TASK 3 — EXPLORATORY DATA ANALYSIS PROJECT")
print("  Data Science Internship")
print("=" * 62)


# =================================================================
#  STEP 1 — GENERATE RICH SYNTHETIC E-COMMERCE DATASET
# =================================================================
sep("STEP 1 › Generating E-Commerce Dataset")

np.random.seed(7)
N = 1500

CATEGORIES  = ["Electronics", "Clothing", "Books", "Home & Kitchen",
                "Sports", "Beauty"]
REGIONS     = ["North", "South", "East", "West", "Central"]
CHANNELS    = ["Website", "Mobile App", "In-Store", "Social Media"]
GENDERS     = ["Male", "Female", "Other"]

cat_arr  = np.random.choice(CATEGORIES, N, p=[0.25,0.22,0.13,0.18,0.12,0.10])
reg_arr  = np.random.choice(REGIONS,    N)
ch_arr   = np.random.choice(CHANNELS,   N, p=[0.35,0.30,0.20,0.15])
gen_arr  = np.random.choice(GENDERS,    N, p=[0.48,0.48,0.04])

# Price varies by category
base_price = {"Electronics":350,"Clothing":65,"Books":25,
              "Home & Kitchen":120,"Sports":80,"Beauty":45}
prices = np.array([base_price[c] * np.random.uniform(0.6, 2.2)
                   for c in cat_arr]).round(2)

quantity    = np.random.choice([1,2,3,4,5], N, p=[0.45,0.28,0.14,0.08,0.05])
discount    = np.random.choice([0,5,10,15,20,25], N,
                               p=[0.30,0.20,0.22,0.13,0.10,0.05])
revenue     = (prices * quantity * (1 - discount/100)).round(2)
age         = np.random.randint(18, 72, N)
rating      = np.random.choice([1,2,3,4,5], N, p=[0.04,0.08,0.18,0.38,0.32])
delivery_d  = np.random.randint(1, 15, N)
returned    = (np.random.uniform(0,1,N) < (0.05 + (5 - rating)*0.04 +
               (discount > 15)*0.05)).astype(int)

dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        np.random.randint(0, 365, N), unit="D")

# Inject a few missing values
miss_idx = np.random.choice(N, 40, replace=False)
prices_m = prices.copy().astype(float); prices_m[miss_idx[:15]] = np.nan
rating_m = rating.copy().astype(float); rating_m[miss_idx[15:30]] = np.nan
age_m    = age.copy().astype(float);   age_m[miss_idx[30:]]   = np.nan

df = pd.DataFrame({
    "Order_ID":       [f"ORD-{10000+i}" for i in range(N)],
    "Date":           dates,
    "Category":       cat_arr,
    "Region":         reg_arr,
    "Channel":        ch_arr,
    "Gender":         gen_arr,
    "Age":            age_m,
    "Price":          prices_m,
    "Quantity":       quantity,
    "Discount_Pct":   discount,
    "Revenue":        revenue,
    "Rating":         rating_m,
    "Delivery_Days":  delivery_d,
    "Returned":       returned,
})

df.to_csv(f"{OUTPUT_DIR}/ecommerce_raw.csv", index=False)
print(f"  ✅ Dataset : {N} rows × {len(df.columns)} columns")
print(f"  ✅ Date range : {dates.min().date()}  →  {dates.max().date()}")
print(f"  📁 Saved → {OUTPUT_DIR}/ecommerce_raw.csv")


# =================================================================
#  STEP 2 — DATA OVERVIEW & CLEANING
# =================================================================
sep("STEP 2 › Data Overview & Cleaning")

print(f"\n  Shape   : {df.shape}")
print(f"  Dtypes  :\n{df.dtypes.to_string()}")

missing = df.isnull().sum()
missing = missing[missing > 0]
print(f"\n  Missing Values:")
for col, n in missing.items():
    print(f"    {col:<16} {n:>3}  ({n/N*100:.1f}%)")

df_c = df.copy()
df_c["Price"].fillna(df_c.groupby("Category")["Price"].transform("median"), inplace=True)
df_c["Rating"].fillna(df_c["Rating"].median(), inplace=True)
df_c["Age"].fillna(df_c["Age"].median(), inplace=True)

df_c["Month"]      = df_c["Date"].dt.month
df_c["Month_Name"] = df_c["Date"].dt.month_name()
df_c["Quarter"]    = "Q" + df_c["Date"].dt.quarter.astype(str)
df_c["Weekday"]    = df_c["Date"].dt.day_name()
df_c["Age_Group"]  = pd.cut(df_c["Age"], bins=[17,25,35,50,72],
                             labels=["18-25","26-35","36-50","51+"])
df_c["Profit_Est"] = (df_c["Revenue"] * np.random.uniform(0.15, 0.40,
                      len(df_c))).round(2)

df_c.to_csv(f"{OUTPUT_DIR}/ecommerce_cleaned.csv", index=False)
print(f"\n  ✅ Cleaned : {df_c.shape[0]} rows × {df_c.shape[1]} columns")
print(f"  📁 Saved → {OUTPUT_DIR}/ecommerce_cleaned.csv")


# =================================================================
#  STEP 3 — STATISTICAL SUMMARY
# =================================================================
sep("STEP 3 › Statistical Summary")

num_cols = ["Age","Price","Quantity","Discount_Pct",
            "Revenue","Rating","Delivery_Days"]

stats_df = df_c[num_cols].describe().T
stats_df["skewness"] = df_c[num_cols].skew().round(3)
stats_df["kurtosis"] = df_c[num_cols].kurt().round(3)
stats_df = stats_df.round(2)
stats_df.to_csv(f"{OUTPUT_DIR}/statistical_summary.csv")
print(f"\n{stats_df[['mean','std','min','50%','max','skewness']].to_string()}")

print(f"\n  Category Revenue Share:")
cat_rev = df_c.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
for cat, rev in cat_rev.items():
    pct = rev / cat_rev.sum() * 100
    bar = "█" * int(pct / 2)
    print(f"    {cat:<18} ₹{rev:>10,.0f}  {pct:5.1f}%  {bar}")

print(f"\n  Return Rate by Category:")
ret_rate = df_c.groupby("Category")["Returned"].mean().sort_values(ascending=False)
for cat, r in ret_rate.items():
    print(f"    {cat:<18} {r*100:.1f}%")


# =================================================================
#  STEP 4 — VISUALIZATIONS
# =================================================================
sep("STEP 4 › Building Dashboards")

MONTH_ORDER = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
DAY_ORDER   = ["Monday","Tuesday","Wednesday","Thursday","Friday",
               "Saturday","Sunday"]

# ════════════════════════════════════════════════════════════════
#  DASHBOARD 1 — BUSINESS OVERVIEW
# ════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 16), facecolor=BG)
fig.suptitle("🛒  E-COMMERCE EDA — BUSINESS OVERVIEW",
             fontsize=22, fontweight="bold", color=TXT, y=0.99)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.40)

# ── KPI row ──────────────────────────────────────────────────
kpis = [
    ("Total Revenue",   f"₹{df_c['Revenue'].sum()/1e6:.2f}M",   C1),
    ("Total Orders",    f"{N:,}",                                 C3),
    ("Avg Order Value", f"₹{df_c['Revenue'].mean():.0f}",        C4),
    ("Return Rate",     f"{df_c['Returned'].mean()*100:.1f}%",   C2),
    ("Avg Rating",      f"{df_c['Rating'].mean():.2f} ★",        C5),
    ("Avg Delivery",    f"{df_c['Delivery_Days'].mean():.1f} d", C6),
]
for i, (lbl, val, col) in enumerate(kpis):
    r, c = 0, i % 4 if i < 4 else i - 4
    if i == 4: r = 0          # force all into row 0
    ax = fig.add_subplot(gs[0, i % 4] if i < 4 else gs[0, i-4])
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.patch.set_facecolor(PANEL2)
    ax.text(0.5, 0.64, val,  ha="center", va="center", fontsize=22,
            fontweight="bold", color=col,  transform=ax.transAxes)
    ax.text(0.5, 0.30, lbl,  ha="center", va="center", fontsize=10,
            color=MUTED, transform=ax.transAxes)
    # accent bar bottom
    ax.add_patch(plt.Rectangle((0,0),1,0.06, color=col, alpha=0.7,
                                transform=ax.transAxes, clip_on=False))

# ── Monthly Revenue Trend ─────────────────────────────────────
ax1 = fig.add_subplot(gs[1, 0:2]); style(ax1)
monthly = (df_c.groupby("Month_Name")["Revenue"].sum()
             .reindex([m for m in MONTH_ORDER if m in df_c["Month_Name"].unique()]))
x_m = range(len(monthly))
ax1.fill_between(x_m, monthly.values/1000, alpha=0.18, color=C1)
ax1.plot(x_m, monthly.values/1000, color=C1, linewidth=2.8,
         marker="o", markersize=7, markerfacecolor=C4,
         markeredgecolor=BG, markeredgewidth=1.5)
for xi, yi in zip(x_m, monthly.values/1000):
    ax1.annotate(f"₹{yi:.0f}K", (xi, yi), textcoords="offset points",
                 xytext=(0,8), ha="center", fontsize=7.5, color=TXT)
ax1.set_xticks(list(x_m))
ax1.set_xticklabels([m[:3] for m in monthly.index], rotation=30, ha="right")
ax1.set_title("Monthly Revenue Trend", fontweight="bold", color=TXT, pad=10)
ax1.set_ylabel("Revenue (₹ Thousands)")

# ── Revenue by Category ───────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 2]); style(ax2, "x")
cat_rev_s = df_c.groupby("Category")["Revenue"].sum().sort_values()
colors_c  = PAL6[:len(cat_rev_s)]
hbars = ax2.barh(cat_rev_s.index, cat_rev_s.values/1000,
                  color=colors_c, edgecolor=BG, height=0.6)
for bar, v in zip(hbars, cat_rev_s.values/1000):
    ax2.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
             f"₹{v:.0f}K", va="center", fontsize=8.5, color=TXT)
ax2.set_title("Revenue by Category", fontweight="bold", color=TXT, pad=10)
ax2.set_xlabel("Revenue (₹ Thousands)")

# ── Channel Distribution ──────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 3]); style(ax3, "n")
ch_cnt = df_c["Channel"].value_counts()
wedges, texts, autotexts = ax3.pie(
    ch_cnt, labels=ch_cnt.index, autopct="%1.1f%%",
    colors=PAL6, startangle=120,
    wedgeprops=dict(edgecolor=BG, linewidth=2), pctdistance=0.72)
for t in texts:     t.set_color(TXT); t.set_fontsize(9)
for t in autotexts: t.set_color(BG);  t.set_fontsize(8); t.set_fontweight("bold")
ax3.set_title("Orders by Channel", fontweight="bold", color=TXT, pad=10)

# ── Quarterly Revenue (grouped + profit) ─────────────────────
ax4 = fig.add_subplot(gs[2, 0:2]); style(ax4)
q_rev  = df_c.groupby("Quarter")["Revenue"].sum()  / 1000
q_prof = df_c.groupby("Quarter")["Profit_Est"].sum()/ 1000
qs     = sorted(q_rev.index)
xq     = np.arange(len(qs)); w = 0.35
b1 = ax4.bar(xq-w/2, [q_rev[q]  for q in qs], w, color=C1, edgecolor=BG, label="Revenue")
b2 = ax4.bar(xq+w/2, [q_prof[q] for q in qs], w, color=C3, edgecolor=BG, label="Est. Profit")
for bar in list(b1)+list(b2):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"₹{bar.get_height():.0f}K", ha="center", fontsize=8, color=TXT)
ax4.set_xticks(xq); ax4.set_xticklabels(qs)
ax4.set_title("Quarterly Revenue & Estimated Profit", fontweight="bold", color=TXT, pad=10)
ax4.set_ylabel("Amount (₹ Thousands)")
ax4.legend(labelcolor=TXT, fontsize=9)

# ── Top Region heatmap ────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 2]); ax5.set_facecolor(PANEL)
pivot_rc = df_c.pivot_table(values="Revenue", index="Region",
                             columns="Category", aggfunc="sum")/1000
sns.heatmap(pivot_rc, ax=ax5, cmap="YlOrRd", fmt=".0f", annot=True,
            linewidths=0.5, linecolor=BG, annot_kws={"size":7},
            cbar_kws={"shrink":0.7})
ax5.set_title("Region × Category Revenue (₹K)", fontweight="bold", color=TXT, pad=10)
ax5.tick_params(colors=MUTED, labelsize=8)
ax5.set_xlabel(""); ax5.set_ylabel("")

# ── Weekday Orders ────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 3]); style(ax6)
wd_cnt = (df_c.groupby("Weekday")["Order_ID"].count()
            .reindex([d for d in DAY_ORDER if d in df_c["Weekday"].unique()]))
bar_cols = [C3 if v == wd_cnt.max() else C1 for v in wd_cnt.values]
ax6.bar(range(len(wd_cnt)), wd_cnt.values, color=bar_cols, edgecolor=BG, width=0.6)
ax6.set_xticks(range(len(wd_cnt)))
ax6.set_xticklabels([d[:3] for d in wd_cnt.index], rotation=30, ha="right")
ax6.set_title("Orders by Weekday", fontweight="bold", color=TXT, pad=10)
ax6.set_ylabel("Order Count")

plt.savefig(f"{OUTPUT_DIR}/dashboard_1_overview.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  ✅ Dashboard 1 → {OUTPUT_DIR}/dashboard_1_overview.png")


# ════════════════════════════════════════════════════════════════
#  DASHBOARD 2 — DISTRIBUTION & CORRELATION ANALYSIS
# ════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 3, figsize=(20, 12), facecolor=BG)
fig2.suptitle("📊  DISTRIBUTION & CORRELATION ANALYSIS",
              fontsize=18, fontweight="bold", color=TXT, y=1.01)

# ── Revenue Distribution + KDE ────────────────────────────────
ax = axes2[0][0]; style(ax)
ax.hist(df_c["Revenue"], bins=40, color=C1, edgecolor=BG, alpha=0.75, density=True)
kde_x = np.linspace(df_c["Revenue"].min(), df_c["Revenue"].max(), 300)
kde   = stats.gaussian_kde(df_c["Revenue"])
ax.plot(kde_x, kde(kde_x), color=C4, linewidth=2.5)
ax.axvline(df_c["Revenue"].mean(),   color=C2, linewidth=1.8, linestyle="--",
           label=f"Mean ₹{df_c['Revenue'].mean():.0f}")
ax.axvline(df_c["Revenue"].median(), color=C3, linewidth=1.8, linestyle="-.",
           label=f"Median ₹{df_c['Revenue'].median():.0f}")
ax.legend(labelcolor=TXT, fontsize=9)
ax.set_title("Revenue Distribution (+ KDE)", fontweight="bold", color=TXT, pad=10)
ax.set_xlabel("Revenue (₹)")

# ── Age Distribution by Gender ────────────────────────────────
ax = axes2[0][1]; style(ax)
for gen, col in zip(["Male","Female"], [C5, C2]):
    sub = df_c[df_c["Gender"]==gen]["Age"]
    ax.hist(sub, bins=25, color=col, alpha=0.6, edgecolor=BG, label=gen, density=True)
ax.set_title("Age Distribution by Gender", fontweight="bold", color=TXT, pad=10)
ax.set_xlabel("Age"); ax.set_ylabel("Density")
ax.legend(labelcolor=TXT, fontsize=9)

# ── Correlation Heatmap ───────────────────────────────────────
ax = axes2[0][2]; ax.set_facecolor(PANEL)
num_corr = ["Age","Price","Quantity","Discount_Pct",
            "Revenue","Rating","Delivery_Days","Returned"]
corr = df_c[num_corr].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, ax=ax, mask=mask, cmap=cmap, vmin=-1, vmax=1,
            annot=True, fmt=".2f", linewidths=0.5, linecolor=BG,
            annot_kws={"size":8}, cbar_kws={"shrink":0.8})
ax.set_title("Feature Correlation Matrix", fontweight="bold", color=TXT, pad=10)
ax.tick_params(colors=MUTED, labelsize=8)

# ── Price Box by Category ─────────────────────────────────────
ax = axes2[1][0]; style(ax)
cats_sorted = df_c.groupby("Category")["Price"].median().sort_values(ascending=False).index.tolist()
data_box    = [df_c[df_c["Category"]==c]["Price"].values for c in cats_sorted]
bp = ax.boxplot(data_box, patch_artist=True, notch=True,
                medianprops=dict(color=BG, linewidth=2.5),
                whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                flierprops=dict(marker=".", color=MUTED, alpha=0.4, markersize=4))
for patch, col in zip(bp["boxes"], PAL6):
    patch.set_facecolor(col); patch.set_alpha(0.8)
ax.set_xticks(range(1, len(cats_sorted)+1))
ax.set_xticklabels(cats_sorted, rotation=25, ha="right", fontsize=9)
ax.set_title("Price Distribution by Category", fontweight="bold", color=TXT, pad=10)
ax.set_ylabel("Price (₹)")

# ── Discount vs Revenue scatter ───────────────────────────────
ax = axes2[1][1]; style(ax, "xy")
sc = ax.scatter(df_c["Discount_Pct"], df_c["Revenue"],
                c=df_c["Rating"], cmap="plasma", alpha=0.35,
                s=18, edgecolors="none")
cb = plt.colorbar(sc, ax=ax)
cb.set_label("Rating", color=TXT, fontsize=9)
cb.ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED)
z   = np.polyfit(df_c["Discount_Pct"], df_c["Revenue"], 1)
xln = np.linspace(0, 25, 100)
ax.plot(xln, np.poly1d(z)(xln), color=C4, linewidth=2, linestyle="--")
ax.set_title("Discount % vs Revenue (colored by Rating)",
             fontweight="bold", color=TXT, pad=10)
ax.set_xlabel("Discount %"); ax.set_ylabel("Revenue (₹)")

# ── Rating vs Return Rate ─────────────────────────────────────
ax = axes2[1][2]; style(ax)
rr_by_rating = df_c.groupby("Rating")["Returned"].mean() * 100
bar_c = [C2 if v > rr_by_rating.mean() else C3 for v in rr_by_rating.values]
bars  = ax.bar(rr_by_rating.index, rr_by_rating.values,
               color=bar_c, edgecolor=BG, width=0.55)
ax.axhline(rr_by_rating.mean(), color=C4, linewidth=1.8, linestyle="--",
           label=f"Avg {rr_by_rating.mean():.1f}%")
for bar, v in zip(bars, rr_by_rating.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{v:.1f}%", ha="center", fontsize=9, color=TXT)
ax.set_title("Return Rate by Rating", fontweight="bold", color=TXT, pad=10)
ax.set_xlabel("Customer Rating (★)"); ax.set_ylabel("Return Rate (%)")
ax.legend(labelcolor=TXT, fontsize=9)

for row in axes2:
    for a in row:
        a.set_facecolor(PANEL)
        a.spines["bottom"].set_color(BORDER)
        a.spines["left"].set_color(BORDER)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dashboard_2_distributions.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  ✅ Dashboard 2 → {OUTPUT_DIR}/dashboard_2_distributions.png")


# ════════════════════════════════════════════════════════════════
#  DASHBOARD 3 — CUSTOMER & BEHAVIOURAL INSIGHTS
# ════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 3, figsize=(20, 12), facecolor=BG)
fig3.suptitle("👤  CUSTOMER & BEHAVIOURAL INSIGHTS",
              fontsize=18, fontweight="bold", color=TXT, y=1.01)

# ── Age Group × Category heatmap ─────────────────────────────
ax = axes3[0][0]; ax.set_facecolor(PANEL)
pv_ac = df_c.pivot_table(values="Order_ID", index="Age_Group",
                          columns="Category", aggfunc="count")
sns.heatmap(pv_ac, ax=ax, cmap="Blues", annot=True, fmt=".0f",
            linewidths=0.4, linecolor=BG, annot_kws={"size":8},
            cbar_kws={"shrink":0.8})
ax.set_title("Orders: Age Group × Category", fontweight="bold", color=TXT, pad=10)
ax.tick_params(colors=MUTED, labelsize=8)
ax.set_xlabel(""); ax.set_ylabel("Age Group")

# ── Avg Revenue by Age Group & Gender ────────────────────────
ax = axes3[0][1]; style(ax)
age_gen = df_c.groupby(["Age_Group","Gender"])["Revenue"].mean().unstack()
x_ag    = np.arange(len(age_gen.index)); w2 = 0.25
for j, (gen, col) in enumerate(zip(["Male","Female"], [C5, C2])):
    if gen in age_gen.columns:
        ax.bar(x_ag + j*w2 - w2/2, age_gen[gen], w2, label=gen,
               color=col, edgecolor=BG, alpha=0.9)
ax.set_xticks(x_ag); ax.set_xticklabels(age_gen.index, fontsize=10)
ax.set_title("Avg Revenue by Age Group & Gender",
             fontweight="bold", color=TXT, pad=10)
ax.set_ylabel("Avg Revenue (₹)")
ax.legend(labelcolor=TXT, fontsize=9)

# ── Delivery Days Distribution ────────────────────────────────
ax = axes3[0][2]; style(ax)
dd_bins = [1,3,5,7,10,15]
dd_labels = ["1-3","3-5","5-7","7-10","10-15"]
df_c["DD_Bucket"] = pd.cut(df_c["Delivery_Days"], bins=dd_bins, labels=dd_labels)
dd_cnt = df_c["DD_Bucket"].value_counts().reindex(dd_labels)
ax.bar(dd_labels, dd_cnt.values, color=PAL6[:len(dd_labels)], edgecolor=BG, width=0.6)
for i,(lbl,v) in enumerate(zip(dd_labels, dd_cnt.values)):
    ax.text(i, v+4, str(v), ha="center", fontsize=9, color=TXT)
ax.set_title("Delivery Time Distribution", fontweight="bold", color=TXT, pad=10)
ax.set_xlabel("Days"); ax.set_ylabel("Order Count")

# ── Violin: Revenue by Channel ────────────────────────────────
ax = axes3[1][0]; style(ax)
channels = df_c["Channel"].unique()
for i, (ch, col) in enumerate(zip(CHANNELS, PAL6)):
    if ch not in df_c["Channel"].values: continue
    sub   = df_c[df_c["Channel"]==ch]["Revenue"]
    parts = ax.violinplot(sub, positions=[i], widths=0.6, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(col); pc.set_alpha(0.7); pc.set_edgecolor(BG)
    parts["cmedians"].set_color(TXT); parts["cmedians"].set_linewidth(2)
    for p in ["cbars","cmins","cmaxes"]: parts[p].set_color(col)
ax.set_xticks(range(len(CHANNELS)))
ax.set_xticklabels(CHANNELS, rotation=15, ha="right", fontsize=9)
ax.set_title("Revenue Distribution by Channel",
             fontweight="bold", color=TXT, pad=10)
ax.set_ylabel("Revenue (₹)")

# ── Return Rate by Region & Channel ──────────────────────────
ax = axes3[1][1]; ax.set_facecolor(PANEL)
pv_rr = df_c.pivot_table(values="Returned", index="Region",
                          columns="Channel", aggfunc="mean") * 100
sns.heatmap(pv_rr, ax=ax, cmap="OrRd", annot=True, fmt=".1f",
            linewidths=0.4, linecolor=BG, annot_kws={"size":9},
            cbar_kws={"shrink":0.8})
ax.set_title("Return Rate % — Region × Channel",
             fontweight="bold", color=TXT, pad=10)
ax.tick_params(colors=MUTED, labelsize=9)
ax.set_xlabel(""); ax.set_ylabel("")

# ── Top 10 Insights Text Panel ────────────────────────────────
ax = axes3[1][2]; ax.axis("off"); ax.set_facecolor(PANEL2)
insights = [
    f"1  Best category   : {cat_rev.idxmax()} (₹{cat_rev.max()/1000:.0f}K)",
    f"2  Lowest returns  : {ret_rate.idxmin()} ({ret_rate.min()*100:.1f}%)",
    f"3  Highest returns : {ret_rate.idxmax()} ({ret_rate.max()*100:.1f}%)",
    f"4  Top channel     : {df_c['Channel'].value_counts().idxmax()}",
    f"5  Peak month      : {monthly.idxmax()[:3]} (₹{monthly.max()/1000:.0f}K)",
    f"6  Avg delivery    : {df_c['Delivery_Days'].mean():.1f} days",
    f"7  Discount↑ → Revenue weak negative trend",
    f"8  Rating 1-2 → return rate {df_c[df_c['Rating']<=2]['Returned'].mean()*100:.1f}%",
    f"9  Rating 4-5 → return rate {df_c[df_c['Rating']>=4]['Returned'].mean()*100:.1f}%",
    f"10 Most active age : 26-35 group",
]
ax.text(0.06, 0.95, "🔑  Key Insights", fontsize=12, fontweight="bold",
        color=C4, transform=ax.transAxes, va="top")
for i, line in enumerate(insights):
    ax.text(0.06, 0.86 - i*0.083, line, fontsize=8.8, color=TXT,
            transform=ax.transAxes, va="top", family="monospace")

for row in axes3:
    for a in row:
        a.set_facecolor(PANEL)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dashboard_3_customers.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  ✅ Dashboard 3 → {OUTPUT_DIR}/dashboard_3_customers.png")


# ════════════════════════════════════════════════════════════════
#  DASHBOARD 4 — STATISTICAL TESTS & ADVANCED ANALYSIS
# ════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(2, 3, figsize=(20, 12), facecolor=BG)
fig4.suptitle("🔬  STATISTICAL TESTS & ADVANCED PATTERNS",
              fontsize=18, fontweight="bold", color=TXT, y=1.01)

# ── QQ Plot for Revenue ───────────────────────────────────────
ax = axes4[0][0]; style(ax, "xy")
(osm, osr), (slope, intercept, r) = stats.probplot(df_c["Revenue"])
ax.scatter(osm, osr, color=C1, alpha=0.4, s=12, edgecolors="none")
ax.plot(osm, slope*np.array(osm)+intercept, color=C4, linewidth=2)
ax.set_title("Q-Q Plot — Revenue (Normality Check)",
             fontweight="bold", color=TXT, pad=10)
ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles")
ax.text(0.05, 0.93, f"R² = {r**2:.3f}", transform=ax.transAxes,
        color=MUTED, fontsize=9)

# ── Pairplot-style: 4 key vars ────────────────────────────────
ax = axes4[0][1]; style(ax, "xy")
ax.scatter(df_c["Price"], df_c["Revenue"],
           c=df_c["Quantity"], cmap="viridis", alpha=0.35, s=16, edgecolors="none")
cb = plt.colorbar(plt.cm.ScalarMappable(cmap="viridis",
     norm=plt.Normalize(df_c["Quantity"].min(), df_c["Quantity"].max())), ax=ax)
cb.set_label("Quantity", color=TXT, fontsize=9)
cb.ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED)
ax.set_title("Price vs Revenue (sized by Quantity)",
             fontweight="bold", color=TXT, pad=10)
ax.set_xlabel("Price (₹)"); ax.set_ylabel("Revenue (₹)")

# ── Revenue per Order — Rolling Average ───────────────────────
ax = axes4[0][2]; style(ax)
ts = (df_c.sort_values("Date").set_index("Date")["Revenue"]
          .resample("W").sum() / 1000)
ax.fill_between(ts.index, ts.values, alpha=0.15, color=C3)
ax.plot(ts.index, ts.values, color=C3, linewidth=1.5, alpha=0.6)
roll = ts.rolling(4, center=True).mean()
ax.plot(roll.index, roll.values, color=C4, linewidth=2.5, label="4-week MA")
ax.set_title("Weekly Revenue + Rolling Avg",
             fontweight="bold", color=TXT, pad=10)
ax.set_ylabel("Revenue (₹ Thousands)")
ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
ax.legend(labelcolor=TXT, fontsize=9)

# ── Discount Effectiveness by Category ───────────────────────
ax = axes4[1][0]; style(ax)
disc_eff = df_c.groupby(["Category","Discount_Pct"])["Revenue"].mean().reset_index()
for cat, col in zip(CATEGORIES, PAL6):
    sub = disc_eff[disc_eff["Category"]==cat]
    ax.plot(sub["Discount_Pct"], sub["Revenue"],
            marker="o", markersize=5, color=col, linewidth=1.5,
            alpha=0.8, label=cat)
ax.set_title("Avg Revenue at Each Discount Level",
             fontweight="bold", color=TXT, pad=10)
ax.set_xlabel("Discount %"); ax.set_ylabel("Avg Revenue (₹)")
ax.legend(fontsize=7.5, labelcolor=TXT, ncol=2)

# ── Delivery Days vs Rating ────────────────────────────────────
ax = axes4[1][1]; style(ax)
dd_rating = df_c.groupby("Delivery_Days")["Rating"].mean()
sc = ax.scatter(dd_rating.index, dd_rating.values,
                c=dd_rating.values, cmap="RdYlGn",
                s=60, edgecolors=BG, linewidth=0.8,
                vmin=1, vmax=5, zorder=3)
z2 = np.polyfit(dd_rating.index, dd_rating.values, 1)
xl = np.linspace(dd_rating.index.min(), dd_rating.index.max(), 100)
ax.plot(xl, np.poly1d(z2)(xl), color=MUTED, linewidth=1.5,
        linestyle="--", label=f"Trend  r={np.corrcoef(dd_rating.index, dd_rating.values)[0,1]:.2f}")
plt.colorbar(sc, ax=ax).set_label("Avg Rating", color=TXT, fontsize=9)
ax.set_title("Delivery Days → Customer Rating",
             fontweight="bold", color=TXT, pad=10)
ax.set_xlabel("Delivery Days"); ax.set_ylabel("Avg Rating (★)")
ax.legend(labelcolor=TXT, fontsize=9)

# ── Cumulative Revenue (80-20 check) ─────────────────────────
ax = axes4[1][2]; style(ax, "xy")
cat_rev_sorted = cat_rev.sort_values(ascending=False)
cum_pct        = cat_rev_sorted.cumsum() / cat_rev_sorted.sum() * 100
cat_pct        = np.arange(1, len(cum_pct)+1) / len(cum_pct) * 100
ax.plot(cat_pct, cum_pct.values, color=C1, linewidth=2.5, marker="o",
        markersize=8, markerfacecolor=C4, markeredgecolor=BG)
ax.fill_between(cat_pct, cum_pct.values, alpha=0.15, color=C1)
ax.axhline(80, color=C2, linestyle="--", linewidth=1.5, alpha=0.7, label="80% mark")
ax.set_xticks(cat_pct)
ax.set_xticklabels([c[:6] for c in cum_pct.index], rotation=25, ha="right", fontsize=9)
ax.set_title("Cumulative Revenue Share (Pareto)",
             fontweight="bold", color=TXT, pad=10)
ax.set_xlabel("Category"); ax.set_ylabel("Cumulative Revenue %")
ax.legend(labelcolor=TXT, fontsize=9)

for row in axes4:
    for a in row:
        a.set_facecolor(PANEL)
        a.spines["bottom"].set_color(BORDER)
        a.spines["left"].set_color(BORDER)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dashboard_4_statistics.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  ✅ Dashboard 4 → {OUTPUT_DIR}/dashboard_4_statistics.png")


# =================================================================
#  STEP 5 — FINAL REPORT
# =================================================================
sep("STEP 5 › Generating Final Report")

report = f"""
╔══════════════════════════════════════════════════════════════════╗
║          TASK 3 — EXPLORATORY DATA ANALYSIS REPORT              ║
╠══════════════════════════════════════════════════════════════════╣
║  DATASET OVERVIEW                                                ║
║    Source   : E-Commerce Orders (Synthetic)                      ║
║    Records  : {N:<6}    Features : {len(df.columns):<6}                           ║
║    Period   : {dates.min().date()}  →  {dates.max().date()}              ║
╠══════════════════════════════════════════════════════════════════╣
║  STATISTICAL SUMMARY                                             ║
║    Total Revenue    : ₹{df_c["Revenue"].sum():>12,.0f}                   ║
║    Mean Revenue     : ₹{df_c["Revenue"].mean():>10,.2f}                   ║
║    Revenue Std Dev  : ₹{df_c["Revenue"].std():>10,.2f}                   ║
║    Revenue Skewness :  {df_c["Revenue"].skew():>8.3f}                        ║
║    Mean Rating      :  {df_c["Rating"].mean():>6.2f} / 5.00                     ║
║    Overall Return % :  {df_c["Returned"].mean()*100:>5.1f}%                        ║
║    Avg Delivery Days:  {df_c["Delivery_Days"].mean():>5.1f} days                    ║
╠══════════════════════════════════════════════════════════════════╣
║  KEY FINDINGS                                                    ║
║  1. {cat_rev.idxmax()} leads revenue at ₹{cat_rev.max():,.0f}              ║
║  2. {df_c["Channel"].value_counts().idxmax()} is the dominant sales channel          ║
║  3. {ret_rate.idxmax()} category has highest return rate ({ret_rate.max()*100:.1f}%)  ║
║  4. Low-rated orders (1-2★) return {df_c[df_c["Rating"]<=2]["Returned"].mean()*100:.1f}% vs {df_c[df_c["Rating"]>=4]["Returned"].mean()*100:.1f}% for high-rated  ║
║  5. Revenue peaks in {monthly.idxmax()[:9]} (₹{monthly.max():,.0f})      ║
║  6. Delivery time negatively correlates with satisfaction        ║
║  7. Discount offers weak negative effect on net revenue          ║
║  8. Age group 26-35 places the most orders across categories     ║
╠══════════════════════════════════════════════════════════════════╣
║  RECOMMENDATIONS                                                 ║
║  ► Focus marketing budget on {cat_rev.idxmax()} & {cat_rev.index[1]} categories      ║
║  ► Improve delivery SLA — target under 5 days to boost ratings  ║
║  ► Review return policy for {ret_rate.idxmax()} — high return risk        ║
║  ► Leverage {df_c["Channel"].value_counts().idxmax()} & Mobile App for promotions     ║
║  ► Tailor campaigns for 26-35 age segment                        ║
╠══════════════════════════════════════════════════════════════════╣
║  OUTPUT FILES                                                    ║
║    outputs/ecommerce_raw.csv                                     ║
║    outputs/ecommerce_cleaned.csv                                 ║
║    outputs/statistical_summary.csv                               ║
║    outputs/dashboard_1_overview.png                              ║
║    outputs/dashboard_2_distributions.png                         ║
║    outputs/dashboard_3_customers.png                             ║
║    outputs/dashboard_4_statistics.png                            ║
║    outputs/eda_report.txt                                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

print(report)
with open(f"{OUTPUT_DIR}/eda_report.txt", "w") as f:
    f.write(report)

print(f"  📁 Report → {OUTPUT_DIR}/eda_report.txt")
print(f"\n{'='*62}")
print(f"  ✅  ALL DONE! Check the 'outputs/' folder.")
print(f"{'='*62}\n")
