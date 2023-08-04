# Cardiovascular_diseases_risk_prediction
Original article: https://eajournals.org/ejcsit/wp-content/uploads/sites/21/2023/06/Integrated-Machine-Learning.pdf

Dataset on Kaggle.com: https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset


# Notes about the original research:
Methods used:  
Logistic Regression 
Model, K-Nearest Neighbor, Naive Bayes, Decision Tree Classifier, and Random Forest Classifier

Questions asked:
1) Which machine learning models were utilized to predict the risk of CVDs based on personal lifestyle 
factors?
2) Among the machine learning models used, which model achieved the best F1 score in predicting 
CVD risk?
3) How did hyperparameter tuning improve the performance of the Logistic Regression model in 
predicting CVD risk?
4) Which personal attributes were identified as having the most impact on predicting the risk of CVDs 
according to the Logistic Regression model?
5) What percentage of people with CVDs and healthy individuals were correctly classified by the 
Logistic Regression model?
6) How well did the Logistic Regression model distinguish between classes, as indicated by the AUC 
score?
7) What insights and significant knowledge can be gained from utilizing personal attributes in predicting the risk of CVDs through machine learning approaches?


# Data Colletion for the Dataset used:
A. Data Collection
The data collection procedure for this study involved utilizing the annual BRFSS data in 2021 obtained 
from the Center for Disease Control (2021). The dataset, which consisted of 438,693 records with a total 
of 304 attributes, was accessed on a local machine. However, not all attributes were relevant to this 
particular study. Consequently, a specific subset of 19 attributes was chosen and incorporated into the 
construction of the machine learning (ML) model to create the predictive model for cardiovascular disease 
(CVD). This deliberate selection led to a reduction in the number of records, resulting in a total of 308,854 
data instances utilized for analysis and model development.

# Models used in the original:
To ensure a balanced evaluation of the models, the F1 score was chosen as the metric, which considers 
both precision and recall. Cross-validation was employed using a 10-Stratified K-Fold, and the mean F1 
score was calculated to represent the overall performance of each model. The ML models used in this 
study included Logistic Regression (LR), Gaussian Naive Bayes (NB), Decision Tree Classifier (DT), K-Nearest Neighbor (KNN), and Random Forest (RF)