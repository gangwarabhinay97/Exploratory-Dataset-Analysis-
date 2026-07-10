import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# Exploratory Data Analysis (EDA) — Iris Dataset

**Internship Project | Data Science**

**Objective:** Analyze the classic Iris dataset to uncover patterns and trends using statistical summaries and visualizations, identify correlations and key influencing factors, and present the findings as a structured report.

**Dataset:** 150 samples of iris flowers across 3 species (*setosa*, *versicolor*, *virginica*), with 4 numeric features: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`.

---
## 1. Setup & Data Loading
""")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("../data/iris.csv")
df.head()
""")

md("## 2. Data Overview\\n\\nCheck shape, data types, and missing values before diving into analysis.")

code("""print("Shape:", df.shape)
df.info()
""")

code("""df.isnull().sum()
""")

md("""**Observation:** The dataset has 150 rows and 5 columns (4 numeric features + 1 categorical target). There are **no missing values**, so no imputation is required.""")

md("## 3. Class Balance\\n\\nChecking how many samples belong to each species.")

code("""df['species'].value_counts()
""")

code("""plt.figure(figsize=(6,4))
sns.countplot(data=df, x='species', hue='species', legend=False)
plt.title('Class Balance: Number of Samples per Species')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
""")

md("""**Observation:** The dataset is perfectly balanced — 50 samples for each of the three species. This is ideal for both statistical comparison and any downstream classification modeling.""")

md("## 4. Statistical Summary\\n\\nOverall descriptive statistics, followed by a breakdown by species.")

code("""df.describe()
""")

code("""df.groupby('species').mean(numeric_only=True)
""")

code("""df.groupby('species').std(numeric_only=True)
""")

md("""**Observations:**
- *Setosa* is clearly the smallest-flowered species on petal dimensions (mean petal length ≈1.46 cm, petal width ≈0.24 cm).
- *Virginica* has the largest petals and sepals on average.
- *Versicolor* sits between the two on almost every measurement.
- Standard deviations show *setosa* is the most tightly clustered species (least internal variation), especially on petal measurements.""")

md("## 5. Correlation Analysis\\n\\nWhich features move together, and which are the strongest predictors of one another?")

code("""corr = df.drop(columns='species').corr(numeric_only=True)
corr
""")

code("""plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={'shrink': .8})
plt.title('Correlation Heatmap of Iris Features')
plt.tight_layout()
plt.show()
""")

md("""**Key findings:**
- `petal_length` and `petal_width` are almost perfectly correlated (**r ≈ 0.96**) — the single strongest relationship in the dataset.
- `sepal_length` correlates strongly with both `petal_length` (**r ≈ 0.87**) and `petal_width` (**r ≈ 0.82**).
- `sepal_width` is only weakly, and *negatively*, correlated with the other three features — it behaves differently from the rest of the measurements and is the least useful feature for distinguishing species on its own.
- **Petal measurements are the key influencing factors** for separating species — far more than sepal measurements.""")

md("## 6. Feature Distributions")

code("""features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, feat in zip(axes.flatten(), features):
    sns.histplot(data=df, x=feat, hue='species', kde=True, ax=ax, element='step')
    ax.set_title(f"Distribution of {feat.replace('_',' ').title()}")
plt.tight_layout()
plt.show()
""")

md("""**Observation:** Petal length and petal width show clean, near-total separation between *setosa* and the other two species, with some overlap remaining between *versicolor* and *virginica*. Sepal width distributions overlap heavily across all three species.""")

md("## 7. Distributions by Species (Boxplots & Violin Plots)")

code("""fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, feat in zip(axes.flatten(), features):
    sns.boxplot(data=df, x='species', y=feat, ax=ax, hue='species', legend=False)
    ax.set_title(f"{feat.replace('_',' ').title()} by Species")
plt.tight_layout()
plt.show()
""")

code("""fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, feat in zip(axes.flatten(), features):
    sns.violinplot(data=df, x='species', y=feat, ax=ax, hue='species', legend=False)
    ax.set_title(f"{feat.replace('_',' ').title()} by Species")
plt.tight_layout()
plt.show()
""")

md("""**Observation:** Boxplots confirm minimal outliers overall. *Setosa* is a clear outlier group (in the statistical sense of being clearly separated) on petal measurements — it never overlaps with the other two species. *Versicolor* and *virginica* show some overlap, mainly on sepal width and sepal length.""")

md("## 8. Pairwise Relationships")

code("""g = sns.pairplot(df, hue='species', diag_kind='hist', height=2.0)
g.fig.suptitle('Pairwise Relationships Between Features by Species', y=1.02)
plt.show()
""")

md("""**Observation:** The pairplot confirms that any scatterplot involving petal length or petal width shows near-perfect linear separation of *setosa* from the other two species, and a good (though imperfect) separation between *versicolor* and *virginica*.""")

md("## 9. Deep Dive: Petal Length vs Petal Width\\n\\nThe strongest correlated pair, and the best predictor of species.")

code("""plt.figure(figsize=(7,5.5))
sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', style='species', s=70)
plt.title('Petal Length vs Petal Width by Species')
plt.tight_layout()
plt.show()
""")

md("""**Observation:** A simple threshold on petal length (~2.5 cm) alone would already separate *setosa* from the other two species with 100% accuracy. Petal width adds the extra separation needed between *versicolor* and *virginica*. This is why petal-based features are typically the top predictors in any Iris species classification model.""")

md("""---
## 10. Summary of Key Insights

1. **No data cleaning was required** — the dataset has zero missing values and is perfectly balanced across the three species (50 samples each).
2. **Petal measurements (length & width) are the strongest influencing factors** for distinguishing species, with a near-perfect correlation of 0.96 between them.
3. **Setosa is easily separable** from the other two species using petal length/width alone — it forms a distinct, non-overlapping cluster.
4. **Versicolor and virginica overlap more**, particularly on sepal measurements, but remain reasonably distinguishable using petal features.
5. **Sepal width is the weakest / most independent feature**, showing weak-to-negative correlation with everything else, and is the least useful single feature for species separation.
6. These findings directly support why petal-based features dominate feature importance in classification models (e.g., Decision Trees, Random Forest, Logistic Regression) trained on this dataset.

## 11. Next Steps
- Build a classification model (Logistic Regression / Decision Tree / Random Forest) using petal features as primary predictors.
- Evaluate feature importance formally using a tree-based model to confirm the correlation-based findings above.
- Consider PCA to visualize the separability in 2D using all four features simultaneously.
""")

nb['cells'] = cells
with open('notebooks/Iris_EDA.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created.")
