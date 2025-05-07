# Credit-Card-Fraud-Detection using Machine Learning

### Overview:
This project tackles the problem of credit card fraud detection using machine learning techniques. Fraudulent transactions are extremely rare (0.17% of the dataset), making this a highly imbalanced classification problem.

We explore both oversampling (SMOTE) and undersampling strategies to handle imbalance, and train multiple models to determine the best performer.

### Dataset:
We used Kaggle - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

Size: 284,807 transactions

Features:
- Time, Amount
- V1 to V28: Principal Components from PCA
- Class: Target (0 = normal, 1 = fraud)

### Exploratory Data Analysis:
- No missing values found in the dataset.

- Highly imbalanced classes:
   - Non-fraud: 284,315
   - Fraud: 492

- Correlation Heatmap shows no strong multicollinearity.


