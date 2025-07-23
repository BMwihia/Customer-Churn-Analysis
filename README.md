## Customer Churn Analysis: End-to-End Data Science Project


![](./images/Comms.jpg)

Image By: [TechFunnel](https://www.pinterest.com/pin/114208540542795588/)



A comprehensive machine learning pipeline to **analyze**, **model**, and **predict customer churn** for a telecom company. The goal is to help the business proactively **identify customers at risk of leaving** and design **data-driven retention strategies**.

---

##  Problem Statement

Customer churn is a major challenge in the telecom industry, leading to **loss of revenue**, **increased acquisition costs**, and **reduced market share**. This project aims to answer:

> **"Can we accurately predict customer churn and identify the key drivers behind it?"**

---

##  Objectives

- Perform Exploratory Data Analysis (EDA) to uncover churn patterns  
- Build and compare multiple classification models  
- Evaluate model performance using metrics like **F1-score**, **Recall**, and **Precision**  
- Apply **GridSearchCV** for hyperparameter tuning  
- Deliver **actionable insights and business recommendations**

---

## Project Structure


- *`README.md` — Project documentation*
- *`notebooks`  — churn_analysis.ipynb - Full EDA, modeling, evaluation*
- *`Images/` — Saved plots and figures*
- *`Data/` — Raw & cleaned datasets*
- *`Presentation/` — Project summary slides PDF*







---

## Dataset Description

The dataset contains **3,333 customer records** and over **20 features**:

| Feature Group       | Key Features |
|---------------------|--------------|
| Demographics        | `state`, `area code`, `phone number` |
| Service Plans       | `international plan`, `voice mail plan` |
| Usage Patterns      | `total day/eve/night/intl minutes`, `calls`, `charges` |
| Interaction Metrics | `number of customer service calls` |
| Target Variable     | `churn` (True/False) |

---

## Data Preparation

- Dropped irrelevant features (e.g., `phone number`)
- Handled missing values (if any)
- Encoded categorical features (`state`, `plans`)
- Standardized numerical features using `StandardScaler`
- Split data using **Stratified Train-Test Split** to preserve class balance

---

## Exploratory Data Analysis (EDA)

**Key Findings:**
- Customers with **international plans** show higher churn rates
- High churn among those with **>3 customer service calls**
- Night and evening minutes have **minimal influence**
- **Class Imbalance**: Only ~14.5% churn → affects model learning

**Visuals Created:**
- Heatmap (correlation matrix)
- Churn distribution bar plot
- KDE plots for continuous features
- Count plots for categorical features

---

## Modeling & Evaluation

We trained and evaluated 4 models using pipelines:

| Model                 | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 0.86     | 0.53      | 0.28   | 0.36     |
| Decision Tree        | 0.93     | 0.82      | 0.63   | 0.71     |
| Random Forest        | 0.91     | 0.95      | 0.38   | 0.54     |
| XGBoost              | **0.95** | **0.90**  | **0.72** | **0.80** |

**XGBoost** outperformed all others in **Recall** and **F1-score**, the key metrics for churn prediction.

---

## Hyperparameter Tuning with GridSearchCV

Used `GridSearchCV` to tune **Random Forest**:

```python
param_grid = {
  'n_estimators': [100, 200],
  'max_depth': [None, 10, 20],
  'min_samples_split': [2, 5],
  'min_samples_leaf': [1, 2]
}
{
  "max_depth": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "n_estimators": 100
}

