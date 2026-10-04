"""
=================================================================
  TASK 2 — PREDICTIVE MODELING USING MACHINE LEARNING
  Data Science Internship
=================================================================
  Models   : Logistic Regression, Decision Tree, Random Forest
  Dataset  : Synthetic Customer Churn Dataset (classification)
  Run with : python ml_predictive_modeling.py
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os

from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier, plot_tree
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics           import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
)
from sklearn.inspection        import permutation_importance

warnings.filterwarnings("ignore")

# ── Output directory ──────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Plot theme ────────────────────────────────────────────────
BG       = "#0F1117"
PANEL    = "#1A1D27"
BORDER   = "#2E3347"
TXT      = "#E2E8F4"
MUTED    = "#8892A4"
ACCENT   = "#7C6AF5"
ACCENT2  = "#F5856A"
ACCENT3  = "#52D9A3"
ACCENT4  = "#F5C842"
PALETTE  = [ACCENT, ACCENT2, ACCENT3, ACCENT4, "#A78BFA", "#F28CB1"]

plt.rcParams.update({
    "figure.facecolor": BG,   "axes.facecolor": PANEL,
    "axes.edgecolor":  BORDER, "axes.labelcolor": TXT,
    "xtick.color": MUTED,      "ytick.color": MUTED,
    "text.color": TXT,         "grid.color": BORDER,
    "grid.linestyle": "--",    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
})

def section(title):
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")

print("=" * 62)
print("  TASK 2 — PREDICTIVE MODELING USING MACHINE LEARNING")
print("  Data Science Internship")
print("=" * 62)


# =================================================================
#  STEP 1 — GENERATE SYNTHETIC CHURN DATASET
# =================================================================
section("STEP 1 › Generating Customer Churn Dataset")

np.random.seed(42)
N = 1000

age            = np.random.randint(18, 70, N)
tenure         = np.random.randint(1, 72, N)          # months
monthly_charge = np.random.uniform(20, 120, N).round(2)
num_products   = np.random.randint(1, 5, N)
support_calls  = np.random.randint(0, 10, N)
has_contract   = np.random.choice([0, 1], N, p=[0.4, 0.6])
payment_method = np.random.choice(["Credit Card","Bank Transfer",
                                    "E-Wallet","Cash"], N)
internet_type  = np.random.choice(["Fiber","DSL","None"], N, p=[0.45,0.40,0.15])

# Churn probability driven by features
churn_prob = (
    0.3
    - 0.003 * tenure
    + 0.004 * support_calls
    + 0.002 * monthly_charge
    - 0.05  * has_contract
    - 0.02  * num_products
    + np.random.normal(0, 0.1, N)
)
churn_prob = np.clip(churn_prob, 0.05, 0.95)
churn      = (np.random.uniform(0, 1, N) < churn_prob).astype(int)

df = pd.DataFrame({
    "Age":            age,
    "Tenure_Months":  tenure,
    "Monthly_Charge": monthly_charge,
    "Num_Products":   num_products,
    "Support_Calls":  support_calls,
    "Has_Contract":   has_contract,
    "Payment_Method": payment_method,
    "Internet_Type":  internet_type,
    "Churn":          churn,
})

df.to_csv(f"{OUTPUT_DIR}/churn_dataset.csv", index=False)
print(f"  ✅ Dataset : {N} rows × {len(df.columns)} columns")
print(f"  ✅ Churn   : {churn.sum()} churned  ({churn.mean()*100:.1f}%)  |  "
      f"{N-churn.sum()} retained")


# =================================================================
#  STEP 2 — PREPROCESSING
# =================================================================
section("STEP 2 › Preprocessing")

df_enc = df.copy()

# Encode categoricals
le = LabelEncoder()
for col in ["Payment_Method", "Internet_Type"]:
    df_enc[col] = le.fit_transform(df_enc[col])

X = df_enc.drop("Churn", axis=1)
y = df_enc["Churn"]

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler  = StandardScaler()
Xtr_sc  = scaler.fit_transform(X_train)
Xte_sc  = scaler.transform(X_test)

print(f"  Train : {len(X_train)} samples    Test : {len(X_test)} samples")
print(f"  Features encoded & scaled with StandardScaler ✅")


# =================================================================
#  STEP 3 — TRAIN MODELS
# =================================================================
section("STEP 3 › Training Models")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=150, max_depth=8,
                                                   random_state=42, n_jobs=-1),
}

results   = {}
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
col_order = ["Accuracy","Precision","Recall","F1","CV Mean","CV Std"]

for name, model in models.items():
    # Use scaled data for LR, raw for tree models
    Xtr = Xtr_sc if name == "Logistic Regression" else X_train.values
    Xte = Xte_sc if name == "Logistic Regression" else X_test.values

    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    rpt     = classification_report(y_test, y_pred, output_dict=True)
    cv_sc   = cross_val_score(model, Xtr, y_train, cv=cv, scoring="accuracy")

    results[name] = {
        "model":    model,
        "y_pred":   y_pred,
        "y_proba":  y_proba,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision":rpt["1"]["precision"],
        "Recall":   rpt["1"]["recall"],
        "F1":       rpt["1"]["f1-score"],
        "CV Mean":  cv_sc.mean(),
        "CV Std":   cv_sc.std(),
        "cm":       confusion_matrix(y_test, y_pred),
    }

    print(f"\n  ► {name}")
    print(f"    Accuracy  : {results[name]['Accuracy']:.4f}")
    print(f"    F1 Score  : {results[name]['F1']:.4f}")
    print(f"    CV (5-fold): {results[name]['CV Mean']:.4f} ± {results[name]['CV Std']:.4f}")

# Metrics table
metrics_df = pd.DataFrame(
    {n: {k: results[n][k] for k in col_order} for n in results}
).T.round(4)
metrics_df.to_csv(f"{OUTPUT_DIR}/model_metrics.csv")
print(f"\n  📁 Metrics saved → {OUTPUT_DIR}/model_metrics.csv")


# =================================================================
#  STEP 4 — VISUALIZATIONS
# =================================================================
section("STEP 4 › Building Dashboards")

# ── Helper ────────────────────────────────────────────────────
def style_ax(ax):
    ax.set_facecolor(PANEL)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.spines["left"].set_color(BORDER)


# ════════════════════════════════════════════════════════════════
#  DASHBOARD 1 — MODEL PERFORMANCE OVERVIEW
# ════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 15), facecolor=BG)
fig.suptitle("🤖  ML MODEL PERFORMANCE DASHBOARD", fontsize=22,
             fontweight="bold", color=TXT, y=0.98)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

model_names  = list(results.keys())
short_names  = ["Log. Reg.", "Dec. Tree", "Rand. Forest"]
model_colors = [ACCENT, ACCENT2, ACCENT3]

# ── KPI Cards ────────────────────────────────────────────────
for i, (name, sname, col) in enumerate(zip(model_names, short_names, model_colors)):
    ax = fig.add_subplot(gs[0, i])
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.set_facecolor("#1E2235")
    ax.patch.set_facecolor("#1E2235")
    acc = results[name]["Accuracy"]
    f1  = results[name]["F1"]
    ax.text(0.5, 0.72, f"{acc*100:.1f}%", ha="center", va="center",
            fontsize=26, fontweight="bold", color=col, transform=ax.transAxes)
    ax.text(0.5, 0.48, sname, ha="center", va="center",
            fontsize=12, color=TXT, transform=ax.transAxes)
    ax.text(0.5, 0.28, f"F1: {f1:.3f}", ha="center", va="center",
            fontsize=10, color=MUTED, transform=ax.transAxes)

# ── Bar Chart — Metric Comparison ────────────────────────────
ax1 = fig.add_subplot(gs[1, 0:2])
style_ax(ax1)
metrics_plot = ["Accuracy", "Precision", "Recall", "F1"]
x   = np.arange(len(metrics_plot))
w   = 0.25
for j, (name, sname, col) in enumerate(zip(model_names, short_names, model_colors)):
    vals = [results[name][m] for m in metrics_plot]
    bars = ax1.bar(x + j*w - w, vals, w, label=sname, color=col,
                   edgecolor=BG, alpha=0.9)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f"{v:.2f}", ha="center", fontsize=7.5, color=TXT)
ax1.set_xticks(x); ax1.set_xticklabels(metrics_plot, fontsize=11)
ax1.set_ylim(0, 1.12); ax1.set_ylabel("Score")
ax1.set_title("Model Comparison — Accuracy · Precision · Recall · F1",
              fontweight="bold", color=TXT, pad=10)
ax1.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TXT, fontsize=9)
ax1.grid(axis="y", alpha=0.4)

# ── CV Box-style bar ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 2])
style_ax(ax2)
cv_means = [results[n]["CV Mean"] for n in model_names]
cv_stds  = [results[n]["CV Std"]  for n in model_names]
bars = ax2.bar(short_names, cv_means, color=model_colors,
               edgecolor=BG, width=0.5, alpha=0.9)
ax2.errorbar(short_names, cv_means, yerr=cv_stds,
             fmt="none", color=TXT, capsize=6, linewidth=2)
ax2.set_ylim(0, 1.1); ax2.set_ylabel("CV Accuracy")
ax2.set_title("5-Fold Cross-Validation", fontweight="bold", color=TXT, pad=10)
for bar, m in zip(bars, cv_means):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
             f"{m:.3f}", ha="center", fontsize=9, color=TXT)
ax2.grid(axis="y", alpha=0.4)

# ── Confusion Matrices (row 2) ────────────────────────────────
for j, (name, sname, col) in enumerate(zip(model_names, short_names, model_colors)):
    ax = fig.add_subplot(gs[2, j])
    ax.set_facecolor(PANEL)
    cm = results[name]["cm"]
    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                cmap=sns.light_palette(col, as_cmap=True),
                linewidths=2, linecolor=BG,
                xticklabels=["No Churn","Churn"],
                yticklabels=["No Churn","Churn"],
                cbar=False, annot_kws={"size": 13, "fontweight": "bold"})
    ax.set_title(f"{sname} — Confusion Matrix",
                 fontweight="bold", color=TXT, pad=8)
    ax.set_xlabel("Predicted", color=MUTED)
    ax.set_ylabel("Actual",    color=MUTED)

plt.savefig(f"{OUTPUT_DIR}/dashboard_1_model_performance.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  ✅ Dashboard 1 → {OUTPUT_DIR}/dashboard_1_model_performance.png")


# ════════════════════════════════════════════════════════════════
#  DASHBOARD 2 — ROC CURVES + FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 2, figsize=(18, 13), facecolor=BG)
fig2.suptitle("📈  ROC CURVES · PRECISION-RECALL · FEATURE IMPORTANCE",
              fontsize=18, fontweight="bold", color=TXT, y=1.01)

# ── ROC Curves ───────────────────────────────────────────────
ax = axes2[0][0]; style_ax(ax)
for name, sname, col in zip(model_names, short_names, model_colors):
    fpr, tpr, _ = roc_curve(y_test, results[name]["y_proba"])
    roc_auc      = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, linewidth=2.5,
            label=f"{sname}  (AUC = {roc_auc:.3f})")
ax.plot([0,1],[0,1], "--", color=MUTED, linewidth=1.2, label="Random Classifier")
ax.fill_between([0,1],[0,1], alpha=0.05, color=MUTED)
ax.set_title("ROC Curves", fontweight="bold", color=TXT, pad=10)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TXT, fontsize=9)
ax.grid(alpha=0.3)

# ── Precision-Recall Curve ────────────────────────────────────
ax = axes2[0][1]; style_ax(ax)
for name, sname, col in zip(model_names, short_names, model_colors):
    prec, rec, _ = precision_recall_curve(y_test, results[name]["y_proba"])
    pr_auc        = auc(rec, prec)
    ax.plot(rec, prec, color=col, linewidth=2.5,
            label=f"{sname}  (AUC = {pr_auc:.3f})")
ax.set_title("Precision-Recall Curves", fontweight="bold", color=TXT, pad=10)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TXT, fontsize=9)
ax.grid(alpha=0.3)

# ── Random Forest Feature Importance ─────────────────────────
ax = axes2[1][0]; style_ax(ax)
rf_model   = results["Random Forest"]["model"]
feat_imp   = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
colors_fi  = [ACCENT if v >= feat_imp.quantile(0.6) else MUTED for v in feat_imp]
bars = ax.barh(feat_imp.index, feat_imp.values, color=colors_fi, edgecolor=BG)
for bar, v in zip(bars, feat_imp.values):
    ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
            f"{v:.3f}", va="center", fontsize=8.5, color=TXT)
ax.set_title("Random Forest — Feature Importance",
             fontweight="bold", color=TXT, pad=10)
ax.set_xlabel("Importance Score")
ax.grid(axis="x", alpha=0.3)

# ── Decision Tree Feature Importance ─────────────────────────
ax = axes2[1][1]; style_ax(ax)
dt_model  = results["Decision Tree"]["model"]
dt_imp    = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values()
colors_dt = [ACCENT2 if v >= dt_imp.quantile(0.6) else MUTED for v in dt_imp]
bars2 = ax.barh(dt_imp.index, dt_imp.values, color=colors_dt, edgecolor=BG)
for bar, v in zip(bars2, dt_imp.values):
    ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
            f"{v:.3f}", va="center", fontsize=8.5, color=TXT)
ax.set_title("Decision Tree — Feature Importance",
             fontweight="bold", color=TXT, pad=10)
ax.set_xlabel("Importance Score")
ax.grid(axis="x", alpha=0.3)

for row in axes2:
    for a in row:
        a.spines["bottom"].set_color(BORDER)
        a.spines["left"].set_color(BORDER)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dashboard_2_roc_features.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  ✅ Dashboard 2 → {OUTPUT_DIR}/dashboard_2_roc_features.png")


# ════════════════════════════════════════════════════════════════
#  DASHBOARD 3 — DATA EDA + DECISION TREE STRUCTURE
# ════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 3, figsize=(20, 12), facecolor=BG)
fig3.suptitle("📊  EXPLORATORY DATA ANALYSIS & DECISION TREE",
              fontsize=18, fontweight="bold", color=TXT, y=1.01)

# ── Churn Distribution ────────────────────────────────────────
ax = axes3[0][0]; style_ax(ax)
churn_counts = y.value_counts()
wedges, texts, autotexts = ax.pie(
    churn_counts, labels=["No Churn","Churn"],
    autopct="%1.1f%%", colors=[ACCENT3, ACCENT2],
    startangle=90, wedgeprops=dict(edgecolor=BG, linewidth=2),
    pctdistance=0.75)
for t in texts:     t.set_color(TXT);  t.set_fontsize(10)
for t in autotexts: t.set_color(BG);   t.set_fontweight("bold")
ax.set_title("Churn Distribution", fontweight="bold", color=TXT, pad=10)

# ── Tenure vs Churn ───────────────────────────────────────────
ax = axes3[0][1]; style_ax(ax)
for val, label, col in [(0,"No Churn",ACCENT3),(1,"Churn",ACCENT2)]:
    sub = df[df["Churn"]==val]["Tenure_Months"]
    ax.hist(sub, bins=25, alpha=0.7, color=col, edgecolor=BG, label=label)
ax.set_title("Tenure Months by Churn", fontweight="bold", color=TXT, pad=10)
ax.set_xlabel("Tenure (Months)"); ax.set_ylabel("Count")
ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TXT, fontsize=9)
ax.grid(axis="y", alpha=0.3)

# ── Monthly Charge vs Churn (violin) ─────────────────────────
ax = axes3[0][2]; style_ax(ax)
for val, label, col, pos in [(0,"No Churn",ACCENT3,-0.2),(1,"Churn",ACCENT2,0.2)]:
    sub = df[df["Churn"]==val]["Monthly_Charge"]
    parts = ax.violinplot(sub, positions=[pos], widths=0.35, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(col); pc.set_alpha(0.7); pc.set_edgecolor(BG)
    parts["cmedians"].set_color(TXT); parts["cmedians"].set_linewidth(2)
    for part in ["cbars","cmins","cmaxes"]:
        parts[part].set_color(col)
ax.set_xticks([-0.2, 0.2]); ax.set_xticklabels(["No Churn","Churn"])
ax.set_title("Monthly Charge Distribution", fontweight="bold", color=TXT, pad=10)
ax.set_ylabel("Monthly Charge (₹)"); ax.grid(axis="y", alpha=0.3)

# ── Support Calls vs Churn Rate ───────────────────────────────
ax = axes3[1][0]; style_ax(ax)
churn_by_calls = df.groupby("Support_Calls")["Churn"].mean() * 100
ax.bar(churn_by_calls.index, churn_by_calls.values,
       color=ACCENT, edgecolor=BG, alpha=0.9)
ax.set_title("Churn Rate by Support Calls", fontweight="bold", color=TXT, pad=10)
ax.set_xlabel("Number of Support Calls"); ax.set_ylabel("Churn Rate (%)")
ax.grid(axis="y", alpha=0.3)

# ── Correlation Heatmap ───────────────────────────────────────
ax = axes3[1][1]; style_ax(ax)
num_cols = ["Age","Tenure_Months","Monthly_Charge","Num_Products",
            "Support_Calls","Has_Contract","Churn"]
corr = df_enc[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=ax, mask=mask, annot=True, fmt=".2f",
            cmap=sns.diverging_palette(240, 10, as_cmap=True),
            vmin=-1, vmax=1, linewidths=0.5, linecolor=BG,
            annot_kws={"size":8}, cbar_kws={"shrink":0.8})
ax.set_title("Feature Correlation Matrix", fontweight="bold", color=TXT, pad=10)
ax.tick_params(colors=MUTED)

# ── Shallow Decision Tree (depth=3) ──────────────────────────
ax = axes3[1][2]; ax.set_facecolor(PANEL); ax.axis("off")
dt_viz = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_viz.fit(X_train, y_train)
plot_tree(dt_viz, ax=ax, feature_names=X.columns.tolist(),
          class_names=["No Churn","Churn"], filled=True,
          impurity=False, rounded=True, proportion=True,
          fontsize=6)
ax.set_title("Decision Tree Structure (depth=3)",
             fontweight="bold", color=TXT, pad=10)

for row in axes3:
    for a in row:
        for sp in ["top","right"]:
            a.spines[sp].set_visible(False)
        a.spines["bottom"].set_color(BORDER)
        a.spines["left"].set_color(BORDER)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dashboard_3_eda_tree.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  ✅ Dashboard 3 → {OUTPUT_DIR}/dashboard_3_eda_tree.png")


# =================================================================
#  STEP 5 — SUMMARY REPORT
# =================================================================
section("STEP 5 › Summary Report")

best_model = max(results, key=lambda n: results[n]["F1"])
best_acc   = results[best_model]["Accuracy"]
best_f1    = results[best_model]["F1"]

report_txt = f"""
╔══════════════════════════════════════════════════════════════╗
║       TASK 2 — PREDICTIVE MODELING REPORT                    ║
╠══════════════════════════════════════════════════════════════╣
║  DATASET                                                     ║
║    Rows    : {N:<6}   Features : {len(X.columns):<6}                          ║
║    Target  : Customer Churn (Binary Classification)          ║
║    Churn % : {churn.mean()*100:.1f}%                                         ║
╠══════════════════════════════════════════════════════════════╣
║  MODEL RESULTS                                               ║
╠══════════════════════════════════════════════════════════════╣"""

for name, sname in zip(model_names, short_names):
    r = results[name]
    report_txt += f"""
║  {sname:<18} Acc={r['Accuracy']:.3f}  F1={r['F1']:.3f}  CV={r['CV Mean']:.3f}±{r['CV Std']:.3f}  ║"""

report_txt += f"""
╠══════════════════════════════════════════════════════════════╣
║  BEST MODEL   : {best_model:<25} F1 = {best_f1:.3f}      ║
║  BEST ACCURACY: {best_acc*100:.2f}%                                          ║
╠══════════════════════════════════════════════════════════════╣
║  TOP FEATURES (Random Forest)                                ║"""

top_feats = pd.Series(rf_model.feature_importances_,
                      index=X.columns).sort_values(ascending=False).head(4)
for feat, imp in top_feats.items():
    report_txt += f"""
║    {feat:<25}  Importance = {imp:.4f}              ║"""

report_txt += f"""
╠══════════════════════════════════════════════════════════════╣
║  OUTPUT FILES                                                ║
║    outputs/churn_dataset.csv                                 ║
║    outputs/model_metrics.csv                                 ║
║    outputs/dashboard_1_model_performance.png                 ║
║    outputs/dashboard_2_roc_features.png                      ║
║    outputs/dashboard_3_eda_tree.png                          ║
╚══════════════════════════════════════════════════════════════╝
"""

print(report_txt)
with open(f"{OUTPUT_DIR}/summary_report.txt", "w") as f:
    f.write(report_txt)

print(f"  📁 Report → {OUTPUT_DIR}/summary_report.txt")
print(f"\n{'='*62}")
print(f"  ✅  ALL DONE!  Check the 'outputs/' folder.")
print(f"{'='*62}\n")
