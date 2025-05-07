# Credit-Card-Fraud-Detection using Machine Learning

### Overview:
This project tackles the problem of credit card fraud detection using machine learning techniques. Fraudulent transactions are extremely rare (0.17% of the dataset), making this a highly imbalanced classification problem.

We explore both oversampling (SMOTE) and undersampling strategies to handle imbalance, and train multiple models to determine the best performer.

### Dataset:
- We used Kaggle - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

- Size: 284,807 transactions

- Features:
  - Time, Amount
  - V1 to V28: Principal Components from PCA
  - Class: Target (0 = normal, 1 = fraud)

### Exploratory Data Analysis:
- No missing values found in the dataset.

- Highly imbalanced classes:
   - Non-fraud: 284,315
   - Fraud: 492

- Correlation Heatmap shows no strong multicollinearity.

### Handling Class Imbalance: 
We compare two main approaches
1. SMOTE (Synthetic Minority Oversampling Technique):
    - Creates synthetic examples of fraud transactions
    - Retains all majority class data
2. Random Undersampling:
    - Reduces non-fraud transactions to match fraud count
    - Faster training, but may lose valuable info

### Models Used:
Trained on both SMOTE and Undersampled data:
Models:
 - Logistic Regression
 - Random Forest Classifier
 - XGBoost

After comparing all, XGBoost with Smote gave the best results.

### Model Optimization:
Performed hyperparameter tuning using GridSearchCV for XGBoost.
param_grid = {
  'n_estimators': [100],
  'max_depth': [5, 7, 9],
  'learning_rate': [0.05, 0.1],
  'subsample': [0.8],
  'colsample_bytree': [0.8]
}

Best Model :
{
  "colsample_bytree": 0.8,
  "learning_rate": 0.1,
  "max_depth": 7,
  "n_estimators": 100,
  "subsample": 0.8
}

### Evaluation Metrics: 
XGBoost + SMOTE Results:

Metrics :
- Precision = 0.79
- Recall = 0.86
- F1-Score = 0.82
- ROC-AUC = 0.98

### PR and ROC Curves: 

![image](https://github.com/user-attachments/assets/448ac0ce-ca88-4da9-9322-3bf861c5e393)

SMOTE shows higher precision at comparable recall levels, confirming its superiority on this dataset.

### Final Conclusion: 
- SMOTE + XGBoost yielded the best fraud detection performance.
- Undersampling gave very high recall but unacceptably low precision.
- GridSearchCV helped fine-tune XGBoost for best results.

#### Requirements 
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn

#### Author
Ankit Soni
B.Tech CSE, IIT Bhilai
Data Science & ML Projects
