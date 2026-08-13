# 🛒 Online Shoppers Purchasing Intention Predictor (End-to-End ML)

An end-to-end Machine Learning pipeline built inside a Jupyter Notebook. It predicts whether an e-commerce website visitor will complete a purchase (`Revenue = True`) or leave without buying (`Revenue = False`), based on real-time browsing session behavioral data.

---

## 📌 Business Overview
In e-commerce, identifying hesitant buyers in real-time allows platforms to trigger targeted automated interventions (e.g., personalized discount pop-ups or live assistant chat) before the user bounces. 

This project implements a clean, production-grade Machine Learning pipeline built with `scikit-learn` to process raw session features, handle data preprocessing, and infer intent efficiently.

---

## 🛠️ Key Features & Engineering Highlights

* **Leakage-Free Preprocessing:** Strict separation of training and testing data before fitting transformers to prevent data leakage.
* **Automated Scikit-Learn Pipeline:** Integrated feature transformation (`ColumnTransformer`) and model inference into a unified `Pipeline` object.
* **Heterogeneous Data Handling:**
  * **Numerical Features:** Imputed via `SimpleImputer(strategy='median')` and feature-scaled using `StandardScaler`.
  * **Categorical Features:** Imputed via `SimpleImputer(strategy='most_frequent')` and encoded using `OneHotEncoder`.
* **Overfitting Prevention:** Applied hyperparameter tuning constraints (`max_depth=4`) on the `DecisionTreeClassifier` to guarantee model generalization on unseen evaluation data.

---

## 📊 Dataset Information

* **Dataset:** [Online Shoppers Purchasing Intention Dataset](https://www.kaggle.com/datasets/rohit2549/online-shoppers-purchasing-intention-dataset)
* **Instances:** 12,330 online browsing sessions
* **Target Feature:** `Revenue` (Binary: `1` = Purchased, `0` = Did Not Purchase)

---

## ⚙️ ML Pipeline Architecture

```text
Raw Data ➔ Train/Test Split ➔ ColumnTransformer ➔ DecisionTree Model ➔ Evaluation
                               ├── Numerical Pipeline (Imputer + Scaler)
                               └── Categorical Pipeline (Imputer + OneHotEncoder)
```
📈 Results & Performance
Test Set Accuracy: ~88%

Overfitting Control: Minimal gap (< 2%) between training and test metrics, confirming strong generalization capacity.


🚀 How to Run Locally

1.Clone this repository:

```Bash
git clone [https://github.com/Mahmoud4265/Online-Shoppers-Purchasing-Intention-Predictor.git](https://github.com/Mahmoud4265/Online-Shoppers-Purchasing-Intention-Predictor.git)
cd Online-Shoppers-Purchasing-Intention-Predictor
```

2.Install dependencies:

```Bash
pip install pandas numpy scikit-learn jupyter
```
3.Launch Jupyter Notebook:

```Bash
jupyter notebook
Open Online-Shoppers-Purchasing-Intention-Predictor.ipynb 
```

📂 Project Structure
```Plaintext
.
├── archive.zip                                                           # Raw Dataset
├── Online-Shoppers-Purchasing-Intention-Predictor.ipynb                  # End-to-End ML Jupyter Notebook
└── README.md                                                             # Project Documentation
```