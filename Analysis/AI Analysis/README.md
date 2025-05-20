# 🔍 AI Analysis – Transformer Oil Age Prediction

This folder contains the core machine learning experiments and evaluations conducted for our project: **Predictive Transformer Health Assessment using a Machine Learning Approach**.

The aim is to determine the **best-performing regression models** to predict:
- **Health Index**  
- **Life Expectancy**  
of transformer oil based on various physicochemical and DGA parameters.

---

## 📁 Contents

### Files

- **AI model.ipynb**  
  Python Jupyter notebook used to:
  - Train/test multiple ML models
  - Evaluate R² scores
  - Compare performance for Health Index and Life Expectancy

- **Health index1.csv**  
  Dataset sourced from Kaggle containing:
  - Dissolved gases
  - Oil quality metrics
  - Output labels: Health Index and Life Expectancy

- **standardized_data.csv**
  The data is standardised via the formula
  *z = (x − μ) ÷ σ*

- **Reg.mlx**
  File made for converting `Health index1.csv` to `standardized_data.csv`

- **RegressionLearnerSession 1.mat** and **RegressionLearnerSession 2.mat**
  Used for checking which model will work using MATLAB R2025a

---

## 📌 Models Evaluated

The following regression models were evaluated in `AI model.ipynb`:
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- CatBoost Regressor
- XGBoost
- Multi-layer Perceptron (MLP)
- K-Nearest Neighbors (KNN)
- Support Vector Regressor (SVR)
- Gaussian Process Regressor
- ElasticNet
- Stacking Regressor
- LightGBM MultiOutput

---

## 📊 Model Performance Comparison

| Model                            | R² (Health Index) | R² (Life Expectancy) |
|----------------------------------|-------------------|-----------------------|
| Multi Task Elastic Net           | 0.735071          | 0.719207              |
| Random Forest Regressor          | 0.893413          | 0.922013              |
| Extra Trees Regressor            | 0.908681          | 0.936439              |
| Gradient Boosting Regressor      | 0.881761          | 0.929379              |
| XGBoost                          | 0.883848          | 0.949412              |
| CatBoost                         | 0.851573          | 0.936672              |
| K-Nearest Neighbors Regressor    | 0.855033          | 0.883082              |
| SVR (Multioutput Regressor)      | 0.745684          | 0.831921              |
| Gaussian Process Regressor       | 0.202147          | 0.326832              |
| MLP Regressor                    | 0.514013          | 0.605714              |
| Regressor Chain                  | 0.476155          | 0.409127              |
| Stacking Regressor               | 0.498361          | 0.556240              |
| LightGBM MultiOutput Regressor   | 0.887267          | 0.928776              |
| Voting Regressor                 | 0.603722          | 0.598618              |

---
## 🏆 Final Selected Models

- **CatBoost Regressor**
- **Extra Trees Regressor**

These two gave the **best performance** in terms of R² and generalization on test data.

---

## 🧪 How to Run

1. Open the Jupyter Notebook:

```bash
jupyter notebook "AI model.ipynb"
