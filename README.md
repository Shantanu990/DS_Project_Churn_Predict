<img width="4342" height="55" alt="image" src="https://github.com/user-attachments/assets/5dfa5d4f-8696-49e4-9206-420ee847a9d1" />**Project Title:** Classification model to predict customer churn and compute churn probabilities

**Project Overview:** An IBM open-source customer churn dataset, sourced from Kaggle, was used for the classification model. With over 7,000 well-structured samples, the dataset was clean and had no missing values, requiring no substantial alteration or formatting. Analysis of the dataset revealed that the churn score failed to provide a definitive estimation of customer churn for approximately 30% of samples where the score ranged between 65 and 80.

**Project Objectives:** 
1. Develop a Classification Model to accurately predict customer churn.
2. The model should also be able to calculate churn probability for each customer.

**Data Source:** https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset

**Exploratory Data Analysis:**
- Random Forest Regressor was used in Python to determine most influential features in generating Churn Value (0 or 1).
- Following encoding/scaling techniques were used for respective features: frequency encoding- City, one hot encoding- Gender, Senior Citizen, Partner, Dependents, Phone Service, Multiple Lines, Internet Service, - Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies, Contract, Paperless Billing, Payment Method, standard scaler- Tenure, Monthly Charges.
- Random Forest Regressor model was trained using features as input and Churn Value as the dependent variable.
- Subsequently, feature_importance_ attribute of scikit-learn library was used to extract relative feature importances.
- Tenure was the most significant contributor followed by Monthly Charges, Internet Service etc.

7,043 samples of telecom service usage are analyzed, revealing that 1,869 customers (26.54%) have churned.Key findings include: 
- Customers on monthly contracts exhibit a disproportionately higher churn rate.
- Customers for whom the dependents or partner status is ‘No’ tend to have a high churn ratio
- Churn  % declines as the tenure increases.
- Churn % exhibits a non-linear dependency on monthly charges. Additionally, a notable outlier range exists where customers with low monthly charges show a disproportionately high churn percentage (highlighted in below image).
- Users who utilize phone service and non-senior citizens have notably higher churn rates.

**Model Development:** 
- A classification model was developed to determine churn probability of customers based on a set of known features. The dataset was prepared with the following feature engineering and encoding techniques:
  1. One Hot Encoding- Gender, Senior Citizen, Partner, Dependents, Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies, Contract, Paperless Billing, Payment Method, standard scaler- Tenure, Monthly Charges.
  2. Frequency Encoding- City
  3. Standard Scaler- Tenure, Monthly Charges
- The dataset was split into an 80:20 ratio for training and testing. Initially Logistic Regression was selected for model training and prediction.
- The model's performance and effectiveness was assessed through following measures:
  1. SHAP Summary and Force Plot: The SHAP summary plot for the LR model revealed an inverse relationship: as monthly charges decreased, the model's predicted churn probability increased, and vice versa. This finding contradicts both the general understanding of customer churn and correlation analysis, which indicated a positive correlation of approximately 0.20 between ‘monthly charges’ and ‘churn’. This suggests a potential learning anomaly in the model's interpretation of this feature, perhaps due to the outlier samples noticed during the EDA.
  2. Precision, Recall, F1-Score and Accuracy: These performance metrics were used to assess overall accuracy and false flags which could cause the telecom company to overspend on customers retention or fail to proactively target customers who are actually going to churn.
  3. ROC AUC: To assess model’s reliability in discriminating between a churner and non-churner.
- Given the Logistic Regression model's misinterpretation of the relationship between monthly charges and churn value, as highlighted by the SHAP summary plot, an XGBoost Classifier was subsequently used for model training, testing, and assessment.

**Tools and libraries used:** 
Software: Python, Power BI, Excel; Libraries: xgboost, sklearn, pandas, numpy, matplotlib, seaborn, shap; Regression models: XGB Classifier, Random Forest, Logistic Regression
 
**Results & Assessment:** 
- SHAP summary/force plots for the LR model revealed an anomalous inverse relationship between monthly charges and churn probability (Figure 3). This learning pattern contradicted both the findings from our correlation analysis (Figure 4) and a direct examination of the dataset. Consequently, an XGBoost Classifier was selected for subsequent model training and evaluation.
- SHAP summary/force plots for the XGB Classifier showed that the model successfully identified and represented the correct, relationship for Monthly Charges: higher charges = higher churn, lower charges = lower churn. This confirms that XGBoost has learned a more accurate and interpretable representation of this feature's impact on churn.
- The initial run of the XGBoost Classifier gave an accuracy of 77%. Upon closer inspection of the predicted values, two distinct patterns of misclassification were identified:
  1. False Negatives for Device Protection: The model frequently predicted a churn value of '0' for customers who actually churned ('1') and had 'Device Protection' enabled. This suggests the model did not adequately capture the relationship between device protection and churn.
  2. False Positives for Specific Demographics: Conversely, the model often predicted a churn value of '1' for customers who did not churn ('0') and had 'Senior Citizen' status as 'No', 'Dependent' status as 'No', and 'Partner' status as 'No'. This indicates the model overemphasized these specific demographic combinations, leading to incorrect churn predictions.
- To address these identified misclassification patterns, sample weights were strategically adjusted. For customers with 'Device Protection' enabled and an actual churn value of '1' (false negatives), their sample weights were increased from 1.0 to 1.2. Conversely, for customers with 'Senior Citizen' status as 'No', 'Dependent' status as 'No', and 'Partner' status as 'No', but an actual churn value of '0' (false positives), their sample weights were increased to 1.15.
- This re-weighting led to a 2% improvement in overall accuracy in the subsequent model run, achieving 79%. Consequently, the XGB Classifier is now capable of providing predictions for the 30% of cases previously lacking definitive estimation (highlighted in problem statement) from the original predictor, with an approximate 80% accuracy.
- However, despite this notable improvement in overall accuracy, the model's ability to precisely predict actual churn (Class 1) still presents a challenge, as reflected in its precision and recall values for this class. This is likely attributable to class imbalance, specifically the limited number of samples available for the churn class (only 1,869 samples). The probability scores for each sample were also determined.

**Next steps for improving the predictive model:**
Following strategies can be used to further improve the classification model:
- **Data Augmentation:** Adding more samples where churn value is 1 will help the model learn feature relationships more accurately.
- **Create synthetic samples:** Create synthetic samples for minority class by using methods such as SMOTE or ADASYN.
- **Hyperparameter tuning:** Use Bayesian Optimization to find better configurations for hyperparameters and contribute to better minority class prediction.

**How to run the model:** Pre-requisites: Python environment- python 3.9+, an IDE like JupyterLab, MS Excel and MS Power BI.
- Download the Telco_customer_churn.xlsx dataset file. Place this file in your python working directory
- To determine feature importance: open the file 'predict_churn.py' in python IDE. Within the py script, locate and run the following code blocks in the specified order: scale_data(), encode(), feature_importance(). A simple plot of the feature importances will be displayed directly in your Python environment's console or plot window.
- To run the classification model: locate and run the following code blocks in 'predict_churn.py': scale_data(), encode(), classifier_xgb() for XGB classifier model or classifier_lr() for logistic regression model. metrics demonstrating model performance, will be displayed in your Python environment's console.
- Running classifier_xgb will also export the results of prediction and probabilities in churn_predictions.csv file in your working directory.
- Optionally run shap_xgb() or shap_lr() to generate SHAP summary plot and force plot for the respective model.
- For a detailed overview of the entire project, including in-depth EDA, model development methodologies, and full results: Navigate to file_name **Churn Classification Model.PDF** in folder 'Project Details'.

**Acknowledgment:** Thankful to Kaggel and user Tanky for publishing the Telco_customer_churn.xlsx dataset
