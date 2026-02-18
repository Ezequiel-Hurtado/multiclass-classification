# Multiclass Classification

## 🧩 Project Overview

This project focuses on predicting the clinical status of patients diagnosed
with Primary Biliary Cirrhosis (PBC) using machine learning techniques.
The goal is to build models capable of estimating the likelihood of different
patient outcomes based on available clinical data.

We work with the well-known Primary Biliary Cirrhosis dataset, which is widely
used in medical and health data science research. The predicted outcomes
correspond to three possible patient statuses: patients who are alive, patients
who are alive after a transplant, and patients who are deceased.

Several classification models are explored, including logistic regression and
XGBoost, with an emphasis on producing reliable probability estimates rather
than only final class predictions. Model hyperparameters are tuned using Optuna
to improve overall performance.

## 📂 Repository Structure

```
📦 project/
│
├── 📁 data/                        # Datasets
│   ├── sample_submission.csv
│   ├── train.csv
│   └── test.csv          
│
├── 📁 scripts/                    
│   ├── EDA.ipynb 
│   ├── Modelling.ipynb            
│   └── functions.py                 
│
└── README.md                   


