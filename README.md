# Adult Income Prediction

Semester project for **Introduction to Machine Learning** (Fall 2026), The University of Lahore.

## Group 5

| Member |
|---|
| Urooj Fatima Muddsar |
| Aayla Zahid |
| Fatima Naveed |

## Project Overview

This project takes one real-world dataset through the complete machine learning lifecycle: data preparation, supervised models, ensembles and tuning, a neural network, unsupervised learning, and a discussion of interpretability and fairness.

- **Dataset:** Adult Census Income Dataset (Kaggle): https://www.kaggle.com/datasets/priyamchoksi/adult-census-income-dataset
- **Size:** 32,561 rows and 15 columns (numeric and categorical features)
- **Task:** Binary classification
- **Target:** `income` (`<=50K` or `>50K`)
- **Success metric:** F1-score (main) and ROC-AUC (supporting)

## Repository Structure

```
adult-income-ml-project/
├── data/
│   └── adult.csv
├── notebooks/
│   └── part1.ipynb
├── .gitignore
└── README.md
```

## Project Progress

| Part | Focus | Status |
|------|-------|--------|
| Part 1 | Data preparation, EDA and leakage-free pipeline | Done |
| Part 2 | Naive Bayes vs decision tree, cross-validation | Planned |
| Part 3 | Ensembles, regularization, hyperparameter tuning | Planned |
| Part 4 | Neural network, clustering/PCA, interpretability and fairness | Planned |
| Part 5 | Final notebook, report and viva | Planned |

## Part 1 Summary

- Converted `?` values to missing values and imputed them inside the pipeline
- Explored the data with four visualisations (class balance, age/hours vs income, education/sex vs income, missing values and skew)
- Handled outliers in `capital.gain` and `capital.loss` with a log transform
- Created new features: `has_capital_gain`, `has_capital_loss`, `is_married`, `age_group`
- Built a leakage-free scikit-learn `Pipeline` with `ColumnTransformer` (fitted on the training set only) and selected the best 40 features with `SelectKBest`
- Handled class imbalance (about 76% / 24%) with a stratified split and class weights

## How to Run

1. Clone the repository.
2. Install the requirements:

```
   pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Open `notebooks/part1.ipynb` in VS Code or Jupyter and click **Run All**.

The notebook reads the data from `../data/adult.csv`, so run it from inside the `notebooks` folder.

## Tools

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter Notebook