# Transformer Oil Age Prediction 🔋

## 📘 Overview

This project was developed as part of our **Introduction to Electrical and Electronics Engineering** end-semester course. The goal of our project is to **predict the aging status of transformer oil**, which is essential for the reliability and safety of power transformers. Transformer oil degrades over time and must be replaced to avoid operational failures.

We created an application that uses **two predictive models**:
- **Life Expectancy Model** – Estimates the remaining useful life of the oil.
- **Health Index Model** – Evaluates the overall health condition of the transformer oil.

By analyzing oil characteristics, we aim to assist engineers in **making data-driven maintenance decisions**.

---

## 🗂️ Project Structure

<pre>
Files
├── Analysis/
│ ├── PCA/
│ │ ├── Analysis.mlx
│ │ └── Health index1.csv
| |
│ └── AI model.ipynb
│ ├── Health index1.csv
│ ├── reg.mlx
│ ├── RegressionLearnerSession 1.mat
│ └── RegressionLearnerSession 2.mat
│ 
├── Final model/
| ├── train.py
| ├── runner.py
| ├── Health index1.csv
| ├── fl1.joblib
| ├── fl2.joblib
│ └── scalar_constants.joblib
└── README.md # Project documentation (this file)
</pre>

---

## 📊 Tools & Technologies

- **Python (Jupyter Notebook)** – Model selection and analysis
- **MATLAB** – PCA, regression training, and data normalization
- **Kaggle Dataset** – Health Index dataset for transformer oil

---

## 🚀 Features

- **Dimensionality Reduction using PCA**
- **Normalization & Standardization**
- **Model comparison** (multiple algorithms evaluated)
- **Trained regression models**
- **Predictive outputs for Life Expectancy and Health Index**

---

## 🧠 Dataset

- **Health index1.csv**: Sourced from Kaggle; contains transformer oil parameters.
- Processed via MATLAB and Python scripts for training and evaluation.

---

## 👨‍💻 Team Members

- Ankur De  
- [Teammate 1's Name]  
- [Teammate 2's Name]  
(*Add or edit names as appropriate.*)

---

## 📌 Notes

- Ensure MATLAB and required toolboxes are installed to run `.mlx` and `.mat` files.
- Python users can run `AI model.ipynb` using standard libraries like `pandas`, `scikit-learn`, etc.
- Final outputs are located in the **Final model** directory.

---

## 📥 License

This project is for academic and educational use. Please credit the authors if reused or modified.

---

## 📞 Contact

For more details, reach out via GitHub or email:
- [Ankur's GitHub Profile](https://github.com/yourusername)  
- [Jiya's Github Profile](https://github.com/JiyaBanglore123)
- [Utham's Github profile](https://github.com/Uk123001)
