# Exploratory Data Analysis Report — Iris Dataset

**Prepared by:** Abhinay Gangwar
**Project type:** Data Science Internship — EDA Assignment

---

## 1. Introduction

This report presents an exploratory data analysis (EDA) of the Iris flower dataset, a widely used dataset in statistics and machine learning containing 150 measurements of iris flowers across three species: *setosa*, *versicolor*, and *virginica*. The goal is to uncover patterns and trends in the data, identify correlations and key influencing factors between features, and summarize the findings in a clear, structured format.

## 2. Dataset Description

| Property | Value |
|---|---|
| Number of records | 150 |
| Number of features | 4 numeric + 1 categorical target |
| Features | `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (all in cm) |
| Target | `species` (setosa, versicolor, virginica) |
| Missing values | None |
| Class balance | 50 samples per species (perfectly balanced) |

## 3. Methodology

The analysis was carried out in the following steps:

1. **Data quality check** — verified data types and confirmed there were no missing or invalid values.
2. **Descriptive statistics** — computed summary statistics (mean, std, min, max, quartiles) overall and grouped by species.
3. **Correlation analysis** — computed a Pearson correlation matrix across all four numeric features.
4. **Univariate visualization** — histograms and KDE plots to examine the distribution of each feature.
5. **Bivariate/multivariate visualization** — boxplots, violin plots, scatter plots, and a full pairplot to examine relationships between features and across species.

## 4. Statistical Summary

Overall, sepal length ranges from 4.3–7.9 cm and petal length from 1.0–6.9 cm, showing that petal measurements have much greater relative spread than sepal measurements. When grouped by species:

- **Setosa** has the smallest and most tightly clustered petal measurements (mean petal length ≈ 1.46 cm, std ≈ 0.17).
- **Versicolor** sits in the middle range across nearly every feature.
- **Virginica** has the largest petal and sepal measurements on average.

This progression (setosa < versicolor < virginica) holds consistently across petal length, petal width, and sepal length, but **not** for sepal width, where setosa actually has the *highest* average value.

## 5. Correlation Analysis

| Feature Pair | Correlation (r) |
|---|---|
| Petal length ↔ Petal width | **0.96** (very strong) |
| Sepal length ↔ Petal length | 0.87 (strong) |
| Sepal length ↔ Petal width | 0.82 (strong) |
| Sepal length ↔ Sepal width | -0.11 (negligible) |
| Sepal width ↔ Petal length | -0.42 (moderate, negative) |
| Sepal width ↔ Petal width | -0.36 (moderate, negative) |

**Key influencing factors:** Petal length and petal width are the two most informative and tightly linked features in the dataset. Sepal length is also a reasonably strong indicator when paired with petal measurements. Sepal width, by contrast, behaves almost independently and is even mildly negatively correlated with the other features — making it the least useful single feature for analysis or prediction.

## 6. Visual Insights

- **Distributions:** Histograms show that petal length and petal width have clearly multi-modal distributions driven by species differences, while sepal width is closer to a single, overlapping normal-like distribution across species.
- **Boxplots/violin plots:** Confirm that setosa is a distinct, non-overlapping cluster on petal measurements, with very few outliers across the dataset overall.
- **Pairplot:** Any pairing that includes petal length or petal width shows strong visual separation between setosa and the other two species. Versicolor and virginica overlap more, particularly on sepal features, but petal-based pairings still separate them reasonably well.
- **Petal length vs. petal width scatter plot:** This is the single most informative 2D view in the dataset — a petal length threshold of roughly 2.5 cm perfectly isolates setosa, and petal width further separates versicolor from virginica.

## 7. Key Findings

1. The dataset is clean, complete, and perfectly balanced, requiring no preprocessing for missing data or class imbalance.
2. Petal measurements (length and width) are the dominant influencing factors for distinguishing between species, far outweighing sepal measurements in discriminative power.
3. Setosa is easily and reliably separable from the other two species using petal features alone.
4. Versicolor and virginica show partial overlap, mainly on sepal-based features, making them somewhat harder to distinguish without petal information.
5. Sepal width is the weakest and most independent feature, contributing the least toward separating species.

## 8. Recommendations / Next Steps

- Use petal length and petal width as the primary features in any downstream classification model (e.g., Logistic Regression, Decision Tree, Random Forest, or k-NN), as they carry the most discriminative signal.
- Formally validate these correlation-based observations using feature importance scores from a trained tree-based model.
- Explore dimensionality reduction (e.g., PCA) to visualize all four features jointly in 2D and confirm species separability.

## 9. Conclusion

This EDA demonstrates that even a small, well-known dataset can yield clear and actionable insights through systematic statistical and visual analysis. The strong correlation between petal length and petal width, combined with clear inter-species separation on these features, explains why petal measurements are consistently the most important predictors in Iris species classification tasks.

---
*Charts referenced in this report are available in the `images/` folder, and the full reproducible analysis is in `notebooks/Iris_EDA.ipynb`.*
