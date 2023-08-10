import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import csv
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
from itertools import combinations 
import scipy.stats as stats
import scipy
# from scipy.stats import ttest_indcou
from scipy.stats import fisher_exact
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
from matplotlib.patches import Patch

# autoreload
# %load_ext autoreload
# %autoreload 2

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ClassPredictionError
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report

cardio = pd.read_csv('data/Cardiovascular_cleaned.csv')
cardio.shape

# This is the Mann-Whitney U Hypothesis Test
# Define the column names and categories
columns = ['Alcohol_Consumption', 'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption']
categories = ['Yes', 'No']

# Define the alpha value
alpha = 0.05

# Loop through each column and perform the Mann-Whitney U test
for col in columns:
    for cat in categories:
        group1 = cardio[cardio['Heart_Disease'] == cat][col]
        group2 = cardio[cardio['Heart_Disease'] != cat][col]

        stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        
        print(f"\nMann-Whitney U test for {col} and Heart_Disease={cat}:")
        print(f"Mann-Whitney U statistic: {stat}")
        print(f"P-value: {p_value}")

        if p_value < alpha:
            print("The difference is statistically significant.")
        else:
            print("There is no significant difference.")

# Mann-Whitney U Hypothesis Test results in a table format with only yes printed
columns = ['Category', 'Heart_Disease', 'Mann_Whitney_U', 'P_Value']
data = [
    ['Alcohol_Consumption', 'Yes', 3006144385.5, 0.0],
    #['Alcohol_Consumption', 'No', 4082698007.5, 0.0],
    ['Fruit_Consumption', 'Yes', 3374577681.0, 2.3590727367213727e-37],
    #['Fruit_Consumption', 'No', 3714264712.0, 2.3590727367213727e-37],
    ['Green_Vegetables_Consumption', 'Yes', 3340434951.0, 3.777408343216292e-52],
    #['Green_Vegetables_Consumption', 'No', 3748407442.0, 3.777408343216292e-52],
    ['FriedPotato_Consumption', 'Yes', 3368543401.5, 1.8633472341292425e-39],
    #['FriedPotato_Consumption', 'No', 3720298991.5, 1.8633472341292425e-39]
]
# Create a DataFrame
result_df = pd.DataFrame(data, columns=columns)
# Display the table
print(result_df)

# This is the beginning of the Logistic regression. I needed to really change the data 

# in the dataframe from catigorical to numerical


# Convert 'Female' and 'Male' in the 'Sex' column to 0 and 1, respectively
cardio['Sex'] = cardio['Sex'].replace({'Female': 0, 'Male': 1})

# Create new binary columns for 'Male' and 'Female'
cardio['Male'] = (cardio['Sex'] == 1).astype(int)
cardio['Female'] = (cardio['Sex'] == 0).astype(int)

# Drop the original 'Sex' column
cardio.drop(columns=['Sex'], inplace=True)

# Verify the counts in 'Male' and 'Female' columns
print(cardio['Male'].value_counts())
print(cardio['Female'].value_counts())

# Map binary categorical variables to 0 and 1
binary_cols = ['Heart_Disease', 'Exercise', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Smoking_History']
cardio[binary_cols] = cardio[binary_cols].replace({'No': 0, 'Yes': 1})

# Apply one-hot encoding to 'General_Health', 'Checkup', and 'Diabetes' columns
cardio = pd.get_dummies(cardio, columns=['General_Health', 'Checkup', 'Diabetes'], drop_first=True)

Average_Age = []
for number in cardio['Age_Category']:
    if '-' in number:
        new = number.split('-')
        Average_Age.append((int(new[0]) + int(new[1])) / 2)
    else: 
        Average_Age.append(85)

# cardio['Average_Age'] = Average_Age

# cardio = cardio.drop(columns = 'Age_Category')

# Due to the last age group being 80+ with no upper limit I guessed and 
# put the highest age at 85. This my impact things at the highest age range


cardio['Average_Age'] = Average_Age
cardio = cardio.drop(columns = 'Age_Category')

'''
This is the Logistic Regression Code

I detailed in the code what my steps at each section were

I am using this doc string to make it easier to find this section

This is also the final copy of all the Regressions I performed with everything in

including spliting the Sex category in to Male and Female. This should give the most 

accurate results for the data analysis. 
'''

# The Logistic Regression Code
# Create new binary columns for 'Male' and 'Female'
# cardio['Male'] = (cardio['Sex'] == 'Male').astype(int)
# cardio['Female'] = (cardio['Sex'] == 'Female').astype(int)

# Define the dependent variable (target) and independent variables (features)
y = cardio['Heart_Disease']  # Dependent variable
X = cardio.drop(columns=['Heart_Disease', 'Sex'])  # Independent variables

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Apply standard scaling to the features to have no variance ##
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the logistic regression model
logit_model = LogisticRegression()

# Train the logistic regression model on the training data
logit_model.fit(X_train, y_train)

# Perform k-fold cross-validation with 10 folds
cv_scores = cross_val_score(logit_model, X_train_scaled, y_train, cv=10, scoring='roc_auc')

# Calculate the average ROC-AUC score
average_roc_auc = np.mean(cv_scores)

# Predict on the test set and calculate ROC-AUC score for testing data. Model evaluated on test data
# threshold = 0.32
y_pred = logit_model.predict_proba(X_test_scaled)[:, 1] # > threshold
roc_auc_test = roc_auc_score(y_test, y_pred)

# Print the results
print("ROC-AUC scores for each fold:")
print(cv_scores)
print("\nAverage ROC-AUC: {:.2f} (±{:.2f})".format(average_roc_auc, np.std(cv_scores)))
print("\nROC-AUC on Testing Data: {:.2f}".format(roc_auc_test))

# Print the classification report on testing data
y_pred_binary = (y_pred >= 0.5).astype(int)
print("\nClassification Report on Testing Data:")
print(classification_report(y_test, y_pred_binary))

# Get feature names from cardio dataset
feature_names = X.columns

# Get feature importances from the fitted logistic regression model
feature_importances = logit_model.coef_[0]

# Sort features by their importance
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_importances = feature_importances[sorted_indices]
sorted_feature_names = feature_names[sorted_indices]

# Remove underscores from feature names
sorted_feature_names = [name.replace('_', ' ') for name in sorted_feature_names]

# Create a list of colors for the bars
colors = ['blue' if imp > 0 else 'red' for imp in sorted_importances]

# Create a horizontal bar chart of feature importances with different colors
plt.figure(figsize=(10, 8))
bars = plt.barh(range(len(sorted_feature_names)), sorted_importances, align='center', color=colors)
plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Plot (Horizontal Bar with Colors)')
plt.tight_layout()

# Create custom legend handles with distinct colors
legend_handles = [Patch(facecolor='blue', edgecolor='black', label='Positive Influence'),
                  Patch(facecolor='red', edgecolor='black', label='Negative Influence')]

# Add the custom legend handles
plt.legend(handles=legend_handles)

plt.show()

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {average_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

if __name__ == "__main__":
    plt.show()

