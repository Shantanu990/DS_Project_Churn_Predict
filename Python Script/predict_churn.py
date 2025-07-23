import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

def scale_data(file_name = 'Telco_customer_churn.xlsx', sheet_name = 'Telco_Churn'):
    df = pd.read_excel(file_name = file_name, sheet_name = sheet_name) # load the dataset to dataframe
    scalar = StandardScaler()
    # scale the continuous numerical values
    df[['Tenure Months', 'Monthly Charges']] = scalar.fit_transform(df[['Tenure Months', 'Monthly Charges']]) 
    return df

def encode(df): # encode all features and define target
    df = pd.get_dummies(df, columns = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method'], drop_first = True) # one-hot encoding suitable for most features
    df['City_encoded'] = df['City'].map(df['City'].value_counts()) # frequency encoding for feature which has too many items 
    features = ['Tenure Months', 'Monthly Charges', 'City_encoded'] + [col for col in df.columns if any(col.startswith(prefix) for prefix in ['Gender','Senior Citizen','Partner','Dependents','Phone Service','Multiple Lines','Internet Service','Online Security','Online Backup','Device Protection','Tech Support','Streaming TV','Streaming Movies','Contract','Paperless Billing','Payment Method'])]
    target = 'Churn Value'
    return features, target

def feature_importance(features, target, df): # determine feature importance in generating churn value
    x = df[features]
    y = df[target]
    model = RandomForestRegressor(random_state = 42) # train using random forest or xgb regressor
    model.fit(x,y)
    importances = pd.Series(model.feature_importances_, index = x.columns)
    importances.sort_values(ascending = False).head(20).plot(kind = 'barh', figsize = (10,8), title = 'Top 20 features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show() # plot top 20 feature categories according to their importance
    importances_df = pd.DataFrame({'feature': x.columns, 'importance': model.feature_importances_})
    importances_df['group'] = importances_df['feature'].str.extract(r'(^[^\_]+)')
    group_importance = importances_df.groupby('group')['importance'].sum().sort_values(ascending=True)
    group_importance.plot(kind='barh', figsize=(8, 6), title='Grouped Feature Importance') # plot feature importance by group
    plt.tight_layout()
    plt.show() # plot feature importance by group

def classifier_lr(features, target, df): # use logistic regression for classifier model
    x = df[features]
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42) # split dataset into subsets for training / testing
    logreg = LogisticRegression(max_iter = 1000, class_weight = 'balanced', random_state=42) # keep class weight as balanced, since samples with churn value as 1 smaller in proportion
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test) # predict churn value for test set
    y_prob = logreg.predict_proba(x_test)[:, 1] # find probability of churn value 1 for each sample 
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC:\n", roc_auc_score(y_test, y_prob))
    return x_train, x_test, logreg

def classifier_xgb(features, target, df): # use xgb classifier for the model
    x = df[features]
    y = df[target]
    customer_ids = df['CustomerID']
    sample_weights = np.ones(len(df))
    cond1 = (df.get('Device Protection_Yes', 0) == 1) & (df['Churn Value'] == 1) 
    cond2 = (df.get('Senior Citizen_Yes', 0)==0) & (df.get('Dependents_Yes', 0)==0) & (df.get('Partner_Yes', 0)==0) & (df['Churn Value']==0)
    sample_weights[cond1] *=1.2 # increase weight of certain features to reduce false +ve and -ve   
    sample_weights[cond2] *=1.15
    x_train, x_test, y_train, y_test, cust_train, cust_test, sw_train, sw_test = train_test_split(x, y, customer_ids, sample_weights, stratify=y, random_state=42) # split dataset into subsets for training / testing
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(x_train, y_train, sample_weight = sw_train)
    y_pred = xgb.predict(x_test)
    y_prob = xgb.predict_proba(x_test)[:, 1] # find probability of churn value 1 for each sample 
    results_df = pd.DataFrame({'Customer ID': cust_test.values, 'Actual': y_test, 'Prediction': y_pred, 'Churn_Proabibility': y_prob})
    # export csv file including details of actual, predicted values and probabilities for each sample in test set
    results_df.to_csv("churn_predictions.csv", index=False)
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC:\n", roc_auc_score(y_test, y_prob))
    return xgb, x_test

def shap_lr(x_train, x_test, logreg): # generate shap summary and values for LR model
    x_train_array = x_train.to_numpy()
    x_test_array = x_test[:500].to_numpy()
    predict_fn = lambda x: logreg.predict_proba(x)[:, 1]
    explainer = shap.KernelExplainer(predict_fn, shap.kmeans(x_train_array, 10)) # kernel explainer required for LR model
    shap_values = explainer.shap_values(x_test_array)
    shap.summary_plot(shap_values, x_test_array, feature_names=x_test.columns) # generate shap summary plot
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0], x_test.iloc[0], feature_names = x_test.columns) # generate shap force plot

def shap_xgb(x_test, xgb): # generate shap summary and values for xgb model
    explainer = shap.TreeExplainer(xgb) 
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, feature_names=x_test.columns) # generate shap summary plot
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0], x_test.iloc[0], feature_names = x_test.columns) # generate shap force plot

def correlation(df): # find correlation between a feature and target value
    print(df[['Monthly Charges', 'Churn Value']].corr())
    sns.boxplot(x = 'Churn Value', y = 'Monthly Charges', data=df)
    plt.show()
