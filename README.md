**Project Title:** Classification model to predict customer churn and compute churn probabilities

**Project Overview:** write up in-progress

**How to run the model:** Pre-requisites: Python environment- python 3.9+, an IDE like JupyterLab, MS Excel and MS Power BI.

- Download the Telco_customer_churn.xlsx dataset file. Place this file in your python working directory
- To determine feature importance: open the file 'predict_churn.py' in python IDE. Within the py script, locate and run the following code blocks in the specified order: scale_data(), encode(), feature_importance(). A simple plot of the feature importances will be displayed directly in your Python environment's console or plot window.
- To run the classification model: locate and run the following code blocks in 'predict_churn.py': scale_data(), encode(), classifier_xgb() for XGB classifier model or classifier_lr() for logistic regression model. metrics demonstrating model performance, will be displayed in your Python environment's console.
- Running classifier_xgb will also export the results of prediction and probabilities in churn_predictions.csv file in your working directory.
- Optionally run shap_xgb() or shap_lr() to generate SHAP summary plot and force plot for the respective model.
- For a detailed overview of the entire project, including in-depth EDA, model development methodologies, and full results: Navigate to file_name (in progress)
Acknowledgment: Thankful to Kaggel and user Tanky for publishing the Telco_customer_churn.xlsx dataset
