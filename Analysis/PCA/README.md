# 🧮 PCA Analysis – Transformer Oil Health Data

This folder contains the **Principal Component Analysis (PCA)** steps and scripts used in our transformer oil age prediction project. PCA was used as a **dimensionality reduction** technique to identify and preserve the most informative features from the dataset.

---

## 🎯 Objective

The goal of PCA here is to:
- Eliminate multicollinearity
- Reduce noise and overfitting
- Improve model performance by projecting data onto **principal components**

---

## 📁 Files Included

| File Name           | Description |
|---------------------|-------------|
| `Analysis.mlx`      | MATLAB Live Script that performs PCA using various methods and visualizes variance explained |
| `Health index1.csv` | Input dataset used for PCA; contains original transformer oil parameters sourced from Kaggle |

---

## 🧪 PCA Methodology

1. **Standardization**: The dataset is normalized to zero mean and unit variance.
2. **Covariance Matrix**: Computed from standardized data.
3. **Eigen Decomposition**: Eigenvalues and eigenvectors are calculated.
4. **Principal Components**: Top components selected based on explained variance.
5. **Projection**: Original data is projected onto the reduced subspace.

---

## 📌 Mehods used for Normalization in Analysis
1. Matlab default
2. By norm
3. By scale - By standard deviation of 1
4. By center - mean
5. By center - median

---

## 🛠️ How to Run (MATLAB Required)

1. Open `Analysis.mlx` in MATLAB.
2. Run all sections to perform PCA.
3. View:
   - Explained variance graph
   - Transformed components
   - Feature contribution plots

---

## 👨‍💻 Contributors

- Ankur De  
- Adina Sree Venkat Utham Kumar  
- Jiya Sachdeva  
- Mamidi Sai Tanushka

---

## 📎 Notes

- PCA is not used for prediction directly but as a preprocessing step.
- Works best when input features are highly correlated.
