import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

def scale_data(file_name = 'Telco_customer_churn.xlsx', sheet_name = 'Telco_Churn'):
    df = pd.read_excel(file_name = file_name, sheet_name = sheet_name)
    scalar = StandardScaler()
    df[['Tenure Months', 'Monthly Charges']] = scalar.fit_transform(df[['Tenure Months', 'Monthly Charges']])
    return df

def encode(df):
    df = pd.get_dummies(df, columns = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method'], drop_first = True)
    df['City_encoded'] = df['City'].map(df['City'].value_counts())
    features = ['Tenure Months', 'Monthly Charges', 'City_encoded'] + [col for col in df.columns if any(col.startswith(prefix) for prefix in ['Gender','Senior Citizen','Partner','Dependents','Phone Service','Multiple Lines','Internet Service','Online Security','Online Backup','Device Protection','Tech Support','Streaming TV','Streaming Movies','Contract','Paperless Billing','Payment Method'])]
    target = 'Churn Value'
    return features, target

def classifier(features, target):
    x = df[features]
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)
    logreg = LogisticRegression(max_iter = 1000, class_weight = 'balanced', random_state=42)
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    y_prob = logreg.predict_proba(x_test)[:, 1]
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC:\n", roc_auc_score(y_test, y_prob))
    return x_train, x_test

def shap(x_train, x_test):
    x_train_array = x_train.to_numpy()
    x_test_array = x_test[:500].to_numpy()
    predict_fn = lambda x: logreg.predict_proba(x)[:, 1]
    explainer = shap.KernelExplainer(predict_fn, shap.kmeans(x_train_array, 10))
    shap_values = explainer.shap_values(x_test_array)
    shap.summary_plot(shap_values, x_test_array, feature_names=x_test.columns)
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0], x_test.iloc[0], feature_names = x_test.columns)