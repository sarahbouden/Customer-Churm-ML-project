# Churn Prediction in the Telecommunications Industry 📈
## Project Overview 🎯
This project implements a complete machine learning pipeline to predict customer churn for a telecommunications company. The primary goal is to identify customers who are likely to discontinue their service, enabling the company to take proactive steps to retain them.

The workflow covers every stage of a data science project, including:

* Data Cleaning & Preprocessing

* Exploratory Data Analysis (EDA)

* Handling Severe Class Imbalance using SMOTE

* Training and evaluating multiple classification models

* Hyperparameter tuning for the best-performing model

The entire analysis is conducted in a Python Jupyter Notebook, leveraging libraries like Scikit-learn, Pandas, and XGBoost.

## The Business Problem: Why Churn Matters 💰
In the highly competitive telecommunications sector, customer churn (or attrition) is a major concern. As highlighted in the reference literature, acquiring a new customer is significantly more expensive than retaining an existing one. High churn rates can lead to:

* Loss of revenue

* Increased marketing costs

* Damage to brand reputation

By accurately predicting churn, a company can target at-risk customers with special offers, improved customer service, or other retention strategies, ultimately protecting its revenue and market share.

## Dataset Used 📊
The project uses a standard telecom churn dataset with the following characteristics:

Total Entries: 7,043 customer records

Features: 21 columns, including:

Customer Demographics: gender, SeniorCitizen, Partner, Dependents

Account Information: tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges

Subscribed Services: PhoneService, MultipleLines, OnlineSecurity, TechSupport, etc.

Target Variable: Churn (Yes/No)

A key challenge identified early on was the class imbalance:

Non-Churners (No): 5,174 (~73.5%)

Churners (Yes): 1,869 (~26.5%)

This imbalance requires special handling to prevent the model from becoming biased towards the majority class.

## Technical Workflow & Methodology ⚙️
### 1. Data Cleaning and Preprocessing
* Handling Missing Values: TotalCharges had a small number of missing values, which were imputed with the column's median.

* Categorical Encoding: LabelEncoder was used for binary categorical features, while OneHotEncoder was applied to features with more than two categories to convert them into a numerical format.

### 2. Exploratory Data Analysis (EDA)
EDA revealed several key insights into the drivers of churn:

* Contract Type: Customers with month-to-month contracts are significantly more likely to churn.

* Tenure: New customers (low tenure) have a much higher churn rate.

* Online Security & Tech Support: Customers without these services tend to churn more often.

### 3. Handling Class Imbalance with SMOTE
To address the ~74/26 class imbalance, we used the Synthetic Minority Over-sampling Technique (SMOTE). SMOTE works by creating new, synthetic data points for the minority class (Churn = 'Yes') based on the feature space of existing minority samples. This balances the dataset and helps the model learn the characteristics of churning customers more effectively.

### 4. Model Building & Evaluation
The dataset was split into training (80%) and testing (20%) sets. Several classification models were trained and evaluated on the SMOTE-resampled training data.

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.76 | 0.65 | 0.71 | 0.68 | 0.75 |
| Support Vector Machine | 0.81 | 0.74 | 0.75 | 0.74 | 0.80 |
| Decision Tree | 0.77 | 0.67 | 0.71 | 0.69 | 0.75 |
| Random Forest | 0.83 | 0.77 | 0.76 | 0.76 | 0.81 |
| **XGBoost Classifier** | **0.85** | **0.78** | **0.80** | **0.79** | **0.84** |

The XGBoost Classifier emerged as the top-performing model across all key metrics, demonstrating a strong balance between precision and recall.

### 5. Hyperparameter Tuning
The XGBoost model's performance was further optimized using GridSearchCV to find the best combination of hyperparameters, leading to a slight improvement in its predictive power.

## Conclusion & Impact 🏁
This project successfully developed a robust machine learning model for predicting customer churn. The XGBoost Classifier, with an F1-Score of 79% and an AUC of 84%, proved to be the most effective algorithm for this task.

By deploying this model, a telecom company can:

* Proactively identify at-risk customers.

* Implement targeted retention campaigns.

* Reduce revenue loss and improve overall customer satisfaction.
