# ACTG 175 Clinical Trial Analysis

A rigorous statistical analysis of the ACTG 175 clinical trial dataset, evaluating the efficacy of Zidovudine (ZDV) and Didanosine (ddI) therapies in adults with HIV.

This project strictly separates machine learning prediction from statistical inference and causal estimation to prevent methodological bias.

## Project Scope

This repository is divided into three distinct analytical tasks:

1. **Prediction (Machine Learning)**
   * **Goal:** Predict the risk of disease progression or death within 24 months (730 days).
   * **Models:** Elastic Net (Post-LASSO), Gradient Boosting (GBM), and Generalized Additive Models (GAM) with B-splines.
   * **Methodology:** Rigorous hyperparameter tuning via GridSearchCV, Stratified 5-Fold Cross-Validation, and log-scale grids to prevent data leakage.

2. **Inference (Survival Analysis)**
   * **Goal:** Estimate the true population probability of remaining event-free at 24 months across four treatment arms.
   * **Models:** Non-parametric Kaplan-Meier estimator.
   * **Methodology:** Exact closed-form estimates utilizing log-log Greenwood 95% Confidence Intervals for robust, mathematically valid bounds.

3. **Causal Inference (G-Computation)**
   * **Goal:** Determine the Average Treatment Effect (ATE) of adding ddI to ZDV therapy.
   * **Models:** Unpenalized Logistic Regression combined with Marginal Standardization.
   * **Methodology:** Used Post-LASSO selected baseline covariates to absorb variance. Applied 1,000 bootstrap iterations to calculate valid 95% Confidence Intervals on the absolute risk reduction.

## Tech Stack

* **Python 3.x**
* **scikit-learn** (Pipelines, LogisticRegression, GradientBoosting, SplineTransformer)
* **statsmodels** (SurvfuncRight / Kaplan-Meier)
* **pandas & numpy** (Data manipulation, Bootstrapping)
* **matplotlib** (Survival curve visualizations)

## App

[Click here for my app](https://mds-aids-project.streamlit.app/)
