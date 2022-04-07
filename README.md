# Predicting Customer Subscription

## Project Description

A bank performed a marketing campaign to track and predict the user behaviour. The marketing campaign was based on phone calls in order to assess if the product (bank term deposit) would be subscribed ('yes') or not subscribed ('no').

The goal of the project is predicting customer subscriptions.

## Project Steps (Notebook Details)

### STEP - 1: EDA & Feature Importance & Feature Engineering

step1_eda_feature_engineering.ipynb

* EDA:
  * Dataset Description
  * Distributions of independent variables
  * Univariate analysis
  * Correlation & Cramers V checks
  * 2 Dimensional Representation of Variables with t-SNE and UMAP
* Feature Selection & Importance:
  * XGBoost, LightGBM, Random Forest Feature Importance Scores
  * LassoCV Coefs
  * Recursive Feature Elimination Rankings
  * Bouta Rankings & Supports
  * Permutation Importance Scores
* Feature Engineering
  * Adding rare classes to most frequent class for categorical features

### STEP - 2: Model Search and Hyperparameter Optimization

#### 2A: step2a_model_search_gridsearch.ipynb

Cross Validation with Grid Search on 80% of the Dataset.
* Models:
  * Logistic Regression
  * KNN
  * Random Forest

#### 2B: step2b_model_search_optuna_xgb_lgb.ipynb

Cross Validation with Bayesian Optimization (Optuna) on 80% of the Dataset.
* Models:
  * LightGBM
  * XGBoost

#### 2C: step2c_model_search_optuna_ann.ipynb

Cross Validation with Bayesian Optimization (Optuna) on 80% of the Dataset.
* Models:
  * Artificial Neural Network

### STEP - 3: Final Test

Final test on test dataset (20% of the total training dataset)
* Models:
  * LightGBM (best model of CV step)
  * XGBoost (best model of CV step)
  * Stacking Ensemble of best LightGBM & XGBoost models
  * Voting Ensembel of best LightGBM & XGBoost models

In this step, Learning Curve Graph of the final model (XGBoost) is checked to control overfitting.

### STEP - 4: Final Predictions

* Training of final model on full training dataset and predicting test dataset.
* Feature Importance Scores and SHAP Values of final model.


## Project Results:

### Best Scores on Cross Validation & Hyperparameter Tuning

Best AUC score for 6 different model after tuning different hyperparameters with several optimization method (GridSearch & Optuna) with different oversampling methods (Random OverSampling, SMOTE, Class Weight Hyperparameter, No Oversampling) are presented below:

| Model | CV Method | Tuning Method | Over Sampling | Categorical Encoding | AUC Score |
| ------ | ------ | ------ | ------ | ------ | ------ |
| `Logistic Regression` | StratifiedKFold | GridSearch | Random | OneHot | 0.9150 |
| `KNN` | StratifiedKFold | GridSearch | Random | OneHot | 0.8337 |
| `Random Forest` | StratifiedKFold | GridSearch | Random | OneHot | 0.9271 |
| `LightGBM` | StratifiedKFold | Optuna | Random | None | 0.9307 |
| `XGBoost` | StratifiedKFold | Optuna | Random | TargetEncoding | `0.9310` |
| `Neural Network` | StratifiedKFold | Optuna | Random | OneHot | 0.9197 |

### Final Test Scores

| Model | AUC Score |
| ------ | ------ |
| `LightGBM` | 0.9278 |
| `XGBoost` | `0.9291` |
| `Neural Network` | 0.9154 |
| `Stacking` | 0.9292 |
| `Voting` | 0.9290 |

Besides Area Under Curve scores, Confusion matrix results of each final model are presented in the notebook.
