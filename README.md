# Cardiovascular_diseases_risk_prediction
Original article: https://eajournals.org/ejcsit/wp-content/uploads/sites/21/2023/06/Integrated-Machine-Learning.pdf

Dataset on Kaggle.com: https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset

# Why This Dataset
Eating healthy has long been linked with a reduction in various ailments. With this dataset I was able to explore whether there is statistical significance between heart disease and Alcohol_Consumption, Fruit_Consumption, Green_Vegatable_Consumption, and FriedPotato_Consumption consuming certain foods. The food columns in the data encompass general categories of food, including alcohol, and are not intended to highlight any specific food or diet in particular. My goal is simply to determine if what is consumed has an impact on a person having heart disease.

# The Data
The dataset use contained 308,854 rows with 19 columns. 

There were 80 duplicate row. I chose to leave the duplicates in since many of the columns were yes/no or some type of answer with a limited range. I expected by chance some of the rows would be the same, and a few were.

The columns are labeled as:
General_Health, Checkup, Exercise, Heart_Disease, Skin_Cancer, Other_Cancer, Depression, Diabetes, Arthritis, Sex, Age_Category, Height(cm), Weight(kg), BMI, Smoking_History, Alcohol_Consumption, Fruit_Consumption, Green_Vegatable_Consumption, and FriedPotato_Consumption

The columns of interest in this research are: Heart_Disease, Alcohol_Consumption, Fruit_Consumption, Green_Vegatable_Consumption, and FriedPotato_Consumption

Below is a graph of the features of interest.
![Histogram of Relavant Columns](images/HistographHypoth.png)


# Hypothesis Test
My hypothesis test is reasonably straght forward:

Null Hypothesis (H0): There is no significant association between the category (alcohol consumption, fruit consumption, green vegetable consumption, fried potato consumption) and the presence of Heart_Disease.

Alternative Hypothesis (Ha): There is a significant association between the category (alcohol consumption, fruit consumption, green vegetable consumption, fried potato consumption) and the presence of Heart_Disease. 

## Testing the Hypothesis
I used the Mann-Whitney U test to check my hypotesis. I checked heart disease yes and no against each of the four columns: alcohol consumption, fruit consumption, green vegetable consumption, fried potato consumption. For the test I used an alpha of 0.05. The results are listed below and are statistically significant with very small P_Values.

                       Category Heart_Disease  Mann_Whitney_U       P_Value
0           Alcohol_Consumption           Yes    3.006144e+09  0.000000e+00
1             Fruit_Consumption           Yes    3.374578e+09  2.359073e-37
2  Green_Vegetables_Consumption           Yes    3.340435e+09  3.777408e-52
3       FriedPotato_Consumption           Yes    3.368543e+09  1.863347e-39

With these results I rejected my Null Hypothesis and accepted my Alternate Hypthesis.

# Logistic Regression
I chose Logistic regression due to Heart Disease being yes/no (binary) instead of continuous.

## How I changed the data
I started with 12 object columns and 7 float columns. I needed to change the types to something more useful.
-I started by changing the Sex column to binary with yes=1 and no-0. 
-Next, I changed the columns General Health , Diabetes, and Checkup from object to binary columns with get-dummies.
-The Age_Category contained two, two digit ages seperated by a hypen, and the oldest age as 80+.
I used split to remove the hypen and took the mean of the two numbers. To romve the + from 80 and best guess for ages in the survey over 80, I put the highest age as 85.
-This left me with integers and floats I could use.

## The Logistic Regression steps

### Splitting the data
I used the train, test, split to seperate the data into training and testing sets.
I set the test size to 0.2 and the random state to 42.

### Scaling
The X_train was scaled, fitted to the model.
The X_test was scaled only.

### The Regression
I used the Logistic Regression with a random state of 42 using the default threshold of 0.5

### Cross-validation
Cross validation was used with 10 K-Folds to train the testing model. Scoring used the 
Receiver Operating Characteristic - Area Under Curve (roc-auc) to evaluate the models performance.

### ROC-AUC
the roc-auc scores were then averaged

### Prediction
The prediction was then run against the test model

# Results of the Logistic Regression
## The Confusion Matrix
![Confusion Matrix](images/ConfusionMatrix.png)

                   Predicted Positive    Predicted Negative
Actual Positive         True Positive           False Negative
Actual Negative         False Positive          True Negative

I this case the Confusion Matrix breakdown explaination is as follows:

Upper Left = 56443 (True - Negative) Instances correctly predicting Negative for Heart Disease

Upper Right = 331 (False Positive) Instances of predicting Heart Disease when actually negative

Lower Left = 4663 (False Negative) Instances of predicting no Heaart Disease when actually had Heart Disease

Lower Right = 334 (True Positive) Instances correctly predicting Positive for Heart Disease








## ROC Curve
![ROC Curve](images/ROC_curve.png)

ROC-AUC scores for each fold:
[0.83581631 0.83996173 0.84231169 0.83711969 0.83089107 0.83413619
 0.83713457 0.8360213  0.82781566 0.82695681]

Average ROC-AUC: 0.83 (±0.00)

ROC-AUC on Testing Data: 0.84

Classification Report on Testing Data:
              precision    recall  f1-score   support

           0       0.92      0.99      0.96     56774
           1       0.50      0.07      0.12      4997

    accuracy                           0.92     61771
   macro avg       0.71      0.53      0.54     61771
weighted avg       0.89      0.92      0.89     61771

## Let's Breakdown the Results
ROC-AUC on Testing Data: 0.83 is the estimate of the model's performance on unseen data or the test set. This model is able to clearly distinguish between Heart Disease = 1 (positive) and Heart Disease = 0 (negative).

### Classification Data
Precision: is the model's ability to correctly predict positives for Heart Disease (= 1) among the true positive cases. This model only predicts 50% of the predicted positives are true positive. Conversely, 92% of predicted negatives are true negatives.

Recall: 0.07 for positive rates indicates this model is only correctly identifying 7% of the actual positive instances.

F1-Score: Is the harmonic mean of precission and recall. This indicates a 12% accuracy for Heart Disease positive cases and a 96% accuracy for negative cases.

Support: This is the number of samples in each class in the testing data. 

Macro Avg: Takes the average for each class without considering the class imbalance.

Weighted Avg: This takes the average for each class weighted by the number of samples in each class.

Remember, the ROC curve is an example of how well the model preformed regardless of the threshold set. In this case it performed well on testing data 84% of the time.













