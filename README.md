# 🍷 Wine Quality Prediction

**A machine learning pipeline for predicting wine quality from physicochemical properties**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

## Overview

Wine quality assessment is traditionally a manual, expert-driven process. This project investigates whether that judgment can be approximated computationally — using measurable physicochemical properties (acidity, sulphates, alcohol content, and more) to predict the quality score a wine would receive.

Three classification models are trained, evaluated, and compared to identify which chemical features most strongly influence perceived quality, and which algorithm generalizes best to unseen data.


![github link](https://github.com/faridah-portfolio/wine-prediction/blob/45a429b55a623560d2ada702b5c1f66b8b2c6d24/FARIDA_Wine_Quality_Prediction.ipynb)


## Models

Three algorithms were selected to represent distinct modeling approaches — an ensemble tree method, a linear model with stochastic optimization, and a kernel-based method — allowing a meaningful comparison of accuracy and robustness.

| Model | Scaling Required | Key Parameters |
|---|---|---|
| Random Forest | No | 200 trees, max_depth = 15 |
| SGD Classifier | Yes | modified_huber loss |
| Support Vector Classifier (SVC) | Yes | RBF kernel |

## Methodology

1. **Data Loading** – load (or simulate) the 1,599-sample red wine dataset, comprising 11 physicochemical features and a quality label
2. **Descriptive Statistics** – summary statistics across all features to establish a baseline understanding of the data
3. **Exploratory Data Analysis** – quality distribution, feature correlation heatmap, alcohol vs. quality relationships, volatile acidity by quality tier, and chemical profile comparisons across quality bands
4. **Preprocessing** – stratified 80/20 train-test split; feature standardization via `StandardScaler`
5. **Model Training** – fit Random Forest, SGD Classifier, and SVC; validate with 5-fold cross-validation
6. **Evaluation** – classification reports, confusion matrices, test vs. cross-validated accuracy, and per-class F1 scores
7. **Feature Importance** – Random Forest importance ranking to identify the strongest quality predictors
8. **Summary & Recommendations** – comparative model selection and suggested next steps

## Key Findings

- **Alcohol content** emerged as the single strongest predictor of wine quality
- **Sulphates** and **volatile acidity** were the next most influential features
- Higher-quality wines are characterized by elevated alcohol and sulphate levels, alongside lower volatile acidity
- Class imbalance — quality scores of 5 and 6 dominate the dataset — constrains model performance on rarer quality classes

## Recommendations for Further Work

- Address class imbalance using SMOTE oversampling or `class_weight='balanced'`
- Reframe the problem as binary classification (low: 3–5 vs. high: 6–8) for a cleaner, more actionable signal
- Apply hyperparameter tuning (`GridSearchCV`) to further improve Random Forest performance
- Evaluate gradient boosting frameworks (XGBoost, LightGBM), which often outperform Random Forest on tabular datasets of this kind

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data Handling | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | scikit-learn (Random Forest, SGD Classifier, SVC) |

## Project Structure

```
├── FARIDA_Wine_Quality_Prediction.ipynb   # Main analysis notebook
└── README.md                              # Project documentation
```

## Author

**Faridat Abubakar**
Data Analyst | Business Analyst
[GitHub](https://github.com/faridah-portfolio)
