# --------------
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Path variable
df = pd.read_csv(path)

# Load the data


# First 5 columns
df.head()

# Independent variables
df.drop('Unnamed: 0',1,inplace = True)
X = df.drop('SeriousDlqin2yrs',1)

# Dependent variable
y = df['SeriousDlqin2yrs']

# Check the value counts
count = df['SeriousDlqin2yrs'].value_counts()

# Split the data set into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 6)




# --------------
# save list of all the columns of X in cols
cols = list(X_train.columns)
# create subplots
fig, axes = plt.subplots(5, 2, sharex='col', sharey='row')

# nested for loops to iterate over all the features and plot the same
for i in range(5):
    for j in range(2):
        col = cols[ i * 2 + j]
        plt.scatter(X_train[col],y_train)
        plt.show()



# --------------
# Check for null values

X_train.isnull().sum()
# Filling the missing values for columns in training data set
X_train.fillna(X_train.median(),inplace = True)

# Filling the missing values for columns in testing data set
X_test.fillna(X_test.median(),inplace = True)

# Checking for null values



# --------------
# Correlation matrix for training set
corr = X_train.corr()


# Plot the heatmap of the correlation matrix
plt.figure(figsize=(15,8))
sns.heatmap(corr,annot = True)
# drop the columns which are correlated amongst each other except one
X_train.drop(['NumberOfTime30-59DaysPastDueNotWorse','NumberOfTime60-89DaysPastDueNotWorse'],1,inplace = True)
X_test.drop(['NumberOfTime30-59DaysPastDueNotWorse','NumberOfTime60-89DaysPastDueNotWorse'],1,inplace = True)


# --------------
from sklearn.preprocessing import StandardScaler
scaler =  StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------------
# Import Logistic regression model and accuracy score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Instantiate the model in a variable in log_reg
log_reg = LogisticRegression()


# Fit the model on training data
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

# Predictions of the training dataset


# accuracy score



# --------------
# Import all the models
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score
# Plot the auc-roc curve
score = roc_auc_score(y_test,y_pred)
y_pred_proba = log_reg.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test,y_pred_proba)
# Evaluation parameters for the model
plt.plot(fpr,tpr,label="Logistic model, auc="+str(auc))
plt.legend(loc=4)
plt.show()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Code starts here
cf= confusion_matrix(y_test, y_pred)
print(cf,'cf = ')
acc = accuracy_score(y_test, y_pred)
print(acc,'acc = ')
precision = precision_score(y_test, y_pred)
print(precision,'pre = ')
recall = recall_score(y_test, y_pred)
print(recall, 'recall = ')
f1 = f1_score(y_test, y_pred)
print(f1, 'fscore = ')



# Code end


# --------------
# Import SMOTE from imblearn library
from imblearn.over_sampling import SMOTE

# Check value counts of target variable for data imbalance
y_train.value_counts()

# Instantiate smote
smote = SMOTE(random_state = 9)

# Fit Smote on training set
X_sample,y_sample = smote.fit_sample(X_train,y_train)
# Check for count of class




# --------------
# Fit logistic regresion model on X_sample and y_sample
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Instantiate the model in a variable in log_reg
log_reg = LogisticRegression()


# Fit the model on training data
log_reg.fit(X_sample,y_sample)
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
y_pred_proba = log_reg.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test,y_pred_proba)
plt.plot(fpr,tpr,label="Logistic model, auc="+str(auc))
plt.legend(loc=4)
plt.show()
# Code end

# Store the result predicted in y_pred

# Store the auc_roc score


# Store the probablity of any class


# Plot the auc_roc_graph


# Print f1_score,Precision_score,recall_score,roc_auc_score and confusion matrix
f= confusion_matrix(y_test, y_pred)
print(cf,'cf = ')
acc = accuracy_score(y_test, y_pred)
print(acc,'acc = ')
precision = precision_score(y_test, y_pred)
print(precision,'pre = ')
recall = recall_score(y_test, y_pred)
print(recall, 'recall = ')
f1 = f1_score(y_test, y_pred)
print(f1, 'fscore = ')


# --------------
# Import RandomForestClassifier from sklearn library
from sklearn.ensemble import RandomForestClassifier

# Instantiate RandomForrestClassifier to a variable rf.
rf=RandomForestClassifier(random_state=9)

# Fit the model on training data.
rf.fit(X_sample, y_sample)

# store the predicted values of testing data in variable y_pred.
y_pred = rf.predict(X_test)
# Store the different evaluation values.
f= confusion_matrix(y_test, y_pred)
print(cf,'cf = ')
acc = accuracy_score(y_test, y_pred)
print(acc,'acc = ')
precision = precision_score(y_test, y_pred)
print(precision,'pre = ')
recall = recall_score(y_test, y_pred)
print(recall, 'recall = ')
f1 = f1_score(y_test, y_pred)
print(f1, 'fscore = ')

# Plot the auc_roc graph
y_pred_proba = rf.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test,y_pred_proba)
plt.plot(fpr,tpr,label="Logistic model, auc="+str(auc))
plt.legend(loc=4)
plt.show()
# Code end


