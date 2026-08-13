<div align="center">

# 🛒 Online Shoppers Purchasing Intention Predictor

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)

*An end-to-end Machine Learning pipeline predicting e-commerce user purchase intent in real-time.*

[Explore Notebook](./Online-Shoppers-Purchasing-Intention-Predictor.ipynb) • [Report Issue](https://github.com/Mahmoud4265/Online-Shoppers-Purchasing-Intention-Predictor/issues)

</div>

---

## 📋 Table of Contents
- [📌 Business Overview](#-business-overview)
- [🛠️ Key Features & Engineering Highlights](#️-key-features--engineering-highlights)
- [⚙️ ML Pipeline Architecture](#️-ml-pipeline-architecture)
- [📊 Dataset Information](#-dataset-information)
- [🚀 How to Run Locally](#-how-to-run-locally)
- [📂 Project Structure](#-project-structure)

---

## 📌 Business Overview
In e-commerce, identifying hesitant buyers in real-time allows platforms to trigger targeted automated interventions (e.g., personalized discount pop-ups or live assistant chat) before the user bounces. 

This project implements a clean, production-grade Machine Learning pipeline built with `scikit-learn` to process raw session features, handle data preprocessing, and infer intent efficiently.

---

## 🛠️ Key Features & Engineering Highlights

* 🛡️ **Leakage-Free Preprocessing:** Strict separation of training and testing data before fitting transformers to prevent data leakage.
* ⚙️ **Automated Scikit-Learn Pipeline:** Integrated feature transformation (`ColumnTransformer`) and model inference into a unified `Pipeline` object.
* 🔄 **Heterogeneous Data Handling:**
  * **Numerical Features:** Imputed via `SimpleImputer(strategy='median')` and feature-scaled using `StandardScaler`.
  * **Categorical Features:** Imputed via `SimpleImputer(strategy='most_frequent')` and encoded using `OneHotEncoder`.
* 🎯 **Overfitting Prevention:** Applied hyperparameter tuning constraints (`max_depth=4`) on the `DecisionTreeClassifier` to guarantee model generalization on unseen evaluation data.

---

## ⚙️ ML Pipeline Architecture

```text
┌──────────┐     ┌──────────────────┐     ┌───────────────────┐     ┌─────────────────────┐     ┌────────────┐
│ Raw Data │ ──> │ Train/Test Split │ ──> │ ColumnTransformer │ ──> │ DecisionTree Model  │ ──> │ Evaluation │
└──────────┘     └──────────────────┘     └─────────┬─────────┘     └─────────────────────┘     └────────────┘
                                                    │
                                                    ├── 📊 Numerical Pipeline (Imputer + Scaler)
                                                    └── 🏷️ Categorical Pipeline (Imputer + OneHot)
```
## 📊 Dataset Information 

- Dataset Source: Kaggle - Online Shoppers Purchasing Intention Dataset
- Instances: 12,330 online browsing sessions
- Target Feature: Revenue (Binary: 1 = Purchased, 0 = Did Not Purchase)


## 🚀 How to Run Locally

1. Clone this repository:
```bash
git clone [https://github.com/Mahmoud4265/Online-Shoppers-Purchasing-Intention-Predictor.git](https://github.com/Mahmoud4265/Online-Shoppers-Purchasing-Intention-Predictor.git)
cd Online-Shoppers-Purchasing-Intention-Predictor
```
2. Install dependencies:
```bash
pip install pandas numpy scikit-learn jupyter
```
3.Launch Jupyter Notebook:
```bash
jupyter notebook
```

Open Online-Shoppers-Purchasing-Intention-Predictor.ipynb and run all cells.

## 📂 Project Structure
```Plaintext
.
├── archive.zip                                           # Raw Dataset
├── Online-Shoppers-Purchasing-Intention-Predictor.ipynb # End-to-End ML Jupyter Notebook
└── README.md                                             # Project Documentation
Crafted with ❤️ for Machine Learning Portfolio
```
