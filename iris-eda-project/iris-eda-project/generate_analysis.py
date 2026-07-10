"""
Generates all statistical summaries and visualizations for the Iris EDA project.
Outputs PNG charts to images/ and a stats summary to report/stats_summary.txt
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 130

df = pd.read_csv("data/iris.csv")

# ---------- 1. Statistical Summary ----------
with open("report/stats_summary.txt", "w") as f:
    f.write("SHAPE\n")
    f.write(str(df.shape) + "\n\n")

    f.write("DTYPES\n")
    f.write(str(df.dtypes) + "\n\n")

    f.write("MISSING VALUES\n")
    f.write(str(df.isnull().sum()) + "\n\n")

    f.write("DESCRIBE (overall)\n")
    f.write(str(df.describe()) + "\n\n")

    f.write("DESCRIBE (by species)\n")
    f.write(str(df.groupby("species").describe().T) + "\n\n")

    f.write("CLASS BALANCE\n")
    f.write(str(df["species"].value_counts()) + "\n\n")

    corr = df.drop(columns="species").corr(numeric_only=True)
    f.write("CORRELATION MATRIX\n")
    f.write(str(corr) + "\n\n")

    # Strongest correlation pair (excluding self-correlation)
    corr_unstacked = corr.where(~np.eye(corr.shape[0], dtype=bool)).unstack().dropna()
    top_corr = corr_unstacked.abs().sort_values(ascending=False).head(3)
    f.write("TOP FEATURE CORRELATIONS\n")
    f.write(str(top_corr) + "\n")

print("Stats summary written.")

# ---------- 2. Pairplot ----------
g = sns.pairplot(df, hue="species", diag_kind="hist", corner=False, height=2.0)
g.fig.suptitle("Pairwise Relationships Between Features by Species", y=1.02, fontsize=14)
g.savefig("images/01_pairplot.png", bbox_inches="tight")
plt.close("all")

# ---------- 3. Correlation Heatmap ----------
plt.figure(figsize=(6, 5))
corr = df.drop(columns="species").corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Heatmap of Iris Features")
plt.tight_layout()
plt.savefig("images/02_correlation_heatmap.png")
plt.close()

# ---------- 4. Boxplots of each feature by species ----------
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, feat in zip(axes.flatten(), features):
    sns.boxplot(data=df, x="species", y=feat, ax=ax, hue="species", legend=False)
    ax.set_title(f"{feat.replace('_', ' ').title()} by Species")
fig.suptitle("Feature Distributions by Species (Boxplots)", fontsize=15)
plt.tight_layout()
plt.savefig("images/03_boxplots_by_species.png")
plt.close()

# ---------- 5. Histograms / distributions ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, feat in zip(axes.flatten(), features):
    sns.histplot(data=df, x=feat, hue="species", kde=True, ax=ax, element="step")
    ax.set_title(f"Distribution of {feat.replace('_', ' ').title()}")
fig.suptitle("Feature Distributions (Histograms + KDE)", fontsize=15)
plt.tight_layout()
plt.savefig("images/04_histograms.png")
plt.close()

# ---------- 6. Violin plots ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, feat in zip(axes.flatten(), features):
    sns.violinplot(data=df, x="species", y=feat, ax=ax, hue="species", legend=False)
    ax.set_title(f"{feat.replace('_', ' ').title()} by Species")
fig.suptitle("Feature Distributions by Species (Violin Plots)", fontsize=15)
plt.tight_layout()
plt.savefig("images/05_violinplots.png")
plt.close()

# ---------- 7. Scatter: petal length vs width (most correlated pair) ----------
plt.figure(figsize=(7, 5.5))
sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species", s=60, style="species")
plt.title("Petal Length vs Petal Width by Species")
plt.tight_layout()
plt.savefig("images/06_petal_scatter.png")
plt.close()

# ---------- 8. Class balance bar chart ----------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="species", hue="species", legend=False)
plt.title("Class Balance: Number of Samples per Species")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("images/07_class_balance.png")
plt.close()

print("All plots generated successfully.")
