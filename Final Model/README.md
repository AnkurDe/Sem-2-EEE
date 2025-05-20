# 🔮 Final Model – Transformer Oil Health & Life Expectancy Prediction

This directory contains the final machine learning pipeline and GUI for predicting:
- **Health Index**
- **Life Expectancy**

using transformer oil test data. This pipeline integrates a **hybrid model** approach for maximum predictive accuracy and includes a graphical user interface (GUI) for usability.

---

## 🧠 Project Overview

This system is designed to:
1. Train optimized regression models using transformer oil data.
2. Save model and normalization artifacts to be reused for inference.
3. Provide an interactive GUI for end-users to input oil test parameters and receive predictions in real time.

---

## 📁 File Descriptions

| File Name             | Description |
|-----------------------|-------------|
| `train.py`            | Trains models on `Health index1.csv`, selects best-performing features, applies preprocessing, saves trained models (`fl1`, `fl2`) and scaler artifacts |
| `runner.py`           | Loads saved models and scaler files to provide a GUI for predictions |
| `Health index1.csv`   | Dataset used for training (sourced from Kaggle, cleaned and filtered) |
| `fl1.joblib`          | Trained ExtraTrees model for Health Index prediction (created by `train.py`) |
| `fl2.joblib`          | Trained CatBoost model for Life Expectancy prediction (created by `train.py`) |
| `scaler.joblib`       | StandardScaler fitted on training features |
| `scalar_constants.joblib` | Stores means and standard deviations of output variables for denormalizing predictions |

---

## ⚙️ Model Details

### 🔹 `train.py`: Model Training Pipeline

#### Preprocessing Steps:
- Drops duplicates and missing values
- Removes outliers beyond ±3 standard deviations
- Applies rolling average smoothing (window=5) to reduce noise
- Standardizes all features using `StandardScaler`

#### Feature Selection:
- Uses **SHAP values** from a MultiOutput ExtraTreesRegressor to identify top 8 most important features:
  - `DBDS`, `Interfacial V`, `Acetylene`, `Methane`, `Power factor`, `Water content`, `CO2`, `Hydrogen`

#### Model Strategy:
| Target            | Model Used         |
|-------------------|--------------------|
| Health Index       | ExtraTreesRegressor |
| Life Expectancy    | CatBoostRegressor (Multi-output) |

- Each model is trained separately to optimize performance.
- Target outputs are normalized during training and denormalized during prediction using saved constants.

#### Artifacts Generated:
- `fl1.joblib`: Health Index model (ExtraTrees)
- `fl2.joblib`: Life Expectancy model (CatBoost)
- `scaler.joblib`: Scaler for feature normalization
- `scalar_constants.joblib`: For reversing normalization on predictions

---

### 🔹 `runner.py`: GUI Application

#### Technologies:
- Built using Python's `tkinter` for UI
- Uses `joblib` to load trained models and scalers
- Accepts 8 input parameters:
  - `DBDS`, `Interfacial V`, `Acetylene`, `Methane`, `Power factor`, `Water content`, `CO2`, `Hydrogen`

#### Workflow:
1. User enters transformer oil test values in GUI
2. Input values are standardized using `scaler.joblib`
3. Health Index is predicted via `fl1.joblib` (ExtraTrees)
4. Life Expectancy is predicted via `fl2.joblib` (CatBoost)
5. Both predictions are denormalized using saved `scalar_constants.joblib`
6. Output is shown in GUI with 2 decimal places

#### GUI Output:
- **Predicted Health Index**
- **Predicted Life Expectancy (in years)**

---

## 💡 Features Importance (Top via SHAP)
- DBDS 5.40
- Interfacial V 3.17
- Acetylene 1.36
- Methane 1.12
- Power factor 0.97
- Water content 0.91
- CO2 0.78
- Hydrogen 0.59



These features contribute most significantly to prediction accuracy.

---

## 🧪 How to Run

### 1. Train the Model (Optional if `.joblib` files already exist)

```bash
python train.py
