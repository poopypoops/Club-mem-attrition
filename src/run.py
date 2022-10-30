#!/usr/bin/env python
# coding: utf-8


import sqlite3
from sqlite3 import Error

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import random
import math


con = sqlite3.connect("data/attrition.db") #lodding the 


df = pd.read_sql_query("SELECT * from attrition", con)



con.close()


# # 2. Feature Engineering

# ## 2.1 Understanding the Data

# - from df.info(), we can see that there are 2470 values, and no null values (no need to replace null values)
# - there are 7 categorical values and 7 numerical values

num_cols = [c for c in df.columns if df[c].dtypes !='O']
cat_cols = [c for c in df.columns if df[c].dtypes == 'O']





# ## 2.2 Cleaning the Data

# ### 2.2.1 Making the Values in the `Qualification` field consistent

df.replace({'Qualification': {"Bachelor's" : 'Bachelor', "Master's" : 'Master', 'Doctor of Philosophy' : "Ph.D"}}, inplace = True)


# ### 2.2.2 Replacing the negative values in the `Age` column 


#calculating the years since the user signed up
def yearSinceSignUp(row):
    year = round(int(row['Months']) / 12)
    return year


df['Time_since_signup_in_years'] = df.apply(yearSinceSignUp, axis=1)


#subtracting the number years since sign up and birth year to get the age the user first signed up
df['Age'] = df.apply(lambda x: (2022- x['Birth Year'] - x['Time_since_signup_in_years']) if x.Age < 0 else x.Age, axis=1)


# ### 2.2.3 Replacing the negative values in the `Birth Year` column

#taking this year - (age first signed up + years as member) to get the birth year
df['Birth Year'] = df.apply(lambda x:  (2022 - (x['Age'] + x['Time_since_signup_in_years'])) if x['Birth Year'] < 0 else x['Birth Year'], axis=1)



df.drop(['Time_since_signup_in_years'], axis=1, inplace = True)


# ### 2.2.4 Making the values in `Travel Time` column consistent



def travel_time_type(row):
    type = row.split(" ")[1]
    value = float(row.split(" ")[0])
    if type == 'hours':
        time = value * 60
        return round(time)
    else:
        return round(value)
        


df['Travel Time'] = df['Travel Time'].apply(travel_time_type)


# ### 2.2.5 Removing negative values in `Montly Income` column




#assumption that there is no negative income (and the negative was an error)
df['Monthly Income'] = df['Monthly Income'].abs()


# ### 2.2.6 Creating new column `Usage`



df['Usage'] = df['Usage Time']* df['Usage Rate']



# ## 2.3 Numerising the Categorical Data

# ### 2.3.1 Changing `Qualification` column to numerical values



def Qualification_to_num(row):
    val_num = np.nan
    x = row['Qualification']
    if x == "Diploma":
        val_num = 1
    elif x == "Bachelor":
        val_num = 2
    elif x == "Master":
        val_num = 3
    elif x == "Ph.D":
        val_num = 4
    return val_num
        
df['Qualification_numerized'] = df.apply(Qualification_to_num, axis=1)


# ### 2.3.2 Changing `Membership` column to numerical values



def Membership_to_num(row):
    val_num = np.nan
    x = row['Membership']
    if x == "Normal":
        val_num = 1
    elif x == "Bronze":
        val_num = 2
    elif x == "Silver":
        val_num = 3
    elif x == "Gold":
        val_num = 4
    return val_num



df['Membership_numerized'] = df.apply(Membership_to_num, axis=1)


# ### 2.3.3 Changing `'Work Domain','Branch','Gender'` column to numerical values



features_category = [
    'Work Domain','Branch','Gender'
]



# Create Dummies Variables for All Categorical Variables

df_dummies = pd.get_dummies(df[features_category], drop_first=True)

# Add the Dummies Variables to the main Data Frame
df = pd.concat([df, df_dummies], axis=1, sort=False)



#dropping irrelevant columns
df_final = df.drop(['Member Unique ID','Gender','Qualification','Branch','Membership','Work Domain'], axis =1)



# # 3. Building the Model


from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn import metrics
import statsmodels.api as sm
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import train_test_split


#defining the features
features = df_final.columns.tolist()



features.remove('Attrition')




#defining the target
target = 'Attrition'




# ## 3.1 Stratified Sampling


df1=df_final[df_final[target]==1]
df1.head()


len(df1)



df0=df_final[df_final[target]==0]


len(df0)


df0_sampled = df0.sample(n=len(df1), random_state=2).copy()


df_new=pd.concat([df1,df0_sampled],axis=0)
df_new[target].value_counts()



# ## 3.2 Spitting the data in to training and testing sets 



# Keep 20% of the data for testing
x_train, x_test, y_train, y_test = model_selection.train_test_split(df_new[features], df_new[target], test_size=0.2, random_state=2)



#scaling the data
scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


# ## 3.3 Logistic Regression Model


# Creating a basic logistic regression model
lg = LogisticRegression() 
# Fit the model to the training set
lg.fit(x_train,y_train.values.ravel())



# print out the model coefficients and intercept
print(lg.coef_)
print(lg.intercept_)


# calculated the training accuracy
lg_base_accuracy_train =  lg.score(x_train, y_train)
lg_base_accuracy_test =  lg.score(x_test, y_test)

print("Training Accuracy: %.2f%%" % (lg_base_accuracy_train*100.0))
print("Testing Accuracy: %.2f%%" % (lg_base_accuracy_test*100.0))




# ### 3.3.1 Evaluating and Improving the Model



# Add a constant to the x_train dataset and rename is as x_train_sm
x_train_sm = sm.add_constant(x_train)




# Create another model using statsmodel's Logit method for model fitting
lg3 = sm.Logit(y_train, x_train_sm).fit()



# Print out the Model Summary
lg3.summary()



df_new[features].columns


#p<0.05 are good indicators - x3,x4





features # ['Travel Time', 'Usage Time']



features_new = ['Travel Time', 'Usage']





x_train_new, x_test_new, y_train_new, y_test_new = model_selection.train_test_split(df_new[features_new], df_new[target], test_size=0.2, random_state=2022)




#scaling the data
scaler = preprocessing.StandardScaler()
x_train_new = scaler.fit_transform(x_train_new)
x_test_new = scaler.fit_transform(x_test_new)




# Creating a basic logistic regression model
lg2 = LogisticRegression() 
# Fit the model to the training set
lg2.fit(x_train_new,y_train_new.values.ravel())





# calculated the training accuracy
lg2_accuracy_train =  lg2.score(x_train_new, y_train_new)
lg2_accuracy_test =  lg2.score(x_test_new, y_test_new)

print("Training Accuracy: %.2f%%" % (lg2_accuracy_train*100.0))
print("Testing Accuracy: %.2f%%" % (lg2_accuracy_test*100.0))


#worst results using the reduced features


# ### 3.3.2 Grid Search



# Determine hyperparameters to auto tune
param_grid = {  'penalty' : ['l1','l2'],
                'C' : np.logspace(-3,3,7),
                'max_iter' : [10,100,1000000],
                'solver' : ['newton-cg', 'lbfgs', 'liblinear'] }





gs = GridSearchCV(lg, param_grid=param_grid, scoring='accuracy', cv= 10, n_jobs=-1)
# cv: number of partitions for cross validation
# n_jobs: number of jobs to run in parallel, -1 means using all processors
gs = gs.fit(x_train, y_train) 

print(gs.best_score_)
print(gs.best_params_)




logreg = LogisticRegression(C = 0.1, 
                            penalty = 'l2', 
                            solver = 'lbfgs',
                           max_iter= 10)
logreg.fit(x_train,y_train)


logreg_accuracy_train =  logreg.score(x_train, y_train)
logreg_accuracy_test =  logreg.score(x_test, y_test)

print("Training Accuracy: %.2f%%" % (logreg_accuracy_train *100.0))
print("Testing Accuracy: %.2f%%" % (logreg_accuracy_test*100.0))


predictions = logreg.predict(x_test)
# Generate Precision of the model
print(metrics.precision_score(y_test, predictions))

# Generate Recall of the model
print(metrics.recall_score(y_test, predictions))





probability_of_above_average = logreg.predict_proba(x_test)
df_proba = pd.DataFrame(probability_of_above_average, columns=['Attrition_false ', 'Attrition_postive'])




df_proba['Prediction'] = predictions





df_proba





# Visualization the Confusion Matrix
sns.heatmap(metrics.confusion_matrix(y_test, predictions)/ len(y_test),
            annot=True, fmt='.2%', 
            xticklabels=['Attrition_negative', 'Attrition_postive'], 
            yticklabels=['Attrition_negative', 'Attrition_postive'])

plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')





# Visualize the distribution of the probability of being positive (above average)
sns.distplot(df_proba['Attrition_postive'])





def precision_and_recal(thresholds):
    for t in thresholds:
        predictions_adjusted = df_proba['Attrition_postive'] > t
        precision = metrics.precision_score(y_test, predictions_adjusted)
        recall = metrics.recall_score(y_test, predictions_adjusted)
        print('At Threshold {:.4f}, Precision = {:.4f} | Recall = {:.4f}'.format(t, precision, recall))
        
precision_and_recal([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])





precision, recall, threshold = metrics.precision_recall_curve(y_test, df_proba['Attrition_postive'])





def plot_precision_recall_vs_threshold(precisions, recalls, thresholds): 
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall") # highlight the threshold and add the legend, axis label, and grid
    plt.legend()
    plt.xlabel('Decision Threshold (Cut-off Point)')
    plt.ylabel('Precision / Recall')
plot_precision_recall_vs_threshold(precision, recall, threshold)


# ### 3.3.3 Saving the model




from joblib import dump, load





# Saving a model
dump(logreg, 'logisticRegression.dump')


# ## 3.4 Decision Tree Model




decision_tree = tree.DecisionTreeClassifier(max_depth = 2)
decision_tree.fit(x_train, y_train)





# calculated the training accuracy
dt_base_accuracy_train =  decision_tree.score(x_train, y_train)
dt_base_accuracy_test =  decision_tree.score(x_test, y_test)


print("Training Accuracy: %.2f%%" % (dt_base_accuracy_train*100.0))
print("Testing Accuracy: %.2f%%" % (dt_base_accuracy_test*100.0))

#print(decision_tree.score(x_train, y_train), '(Train Accuracy)')
#print(decision_tree.score(x_test, y_test), '(Test Accuracy)')


# ### 3.4.1 Boosting Decision Tree




from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=8,learning_rate=0.6)
adb.fit(x_train, y_train)





adb_base_accuracy_train =  adb.score(x_train, y_train)
adb_base_accuracy_test =  adb.score(x_test, y_test)


print("Training Accuracy: %.2f%%" % (adb_base_accuracy_train*100.0))
print("Testing Accuracy: %.2f%%" % (adb_base_accuracy_test*100.0))

#print(decision_tree.score(x_train, y_train), '(Train Accuracy)')
#print(decision_tree.score(x_test, y_test), '(Test Accuracy)')


# ### 3.4.2 Bagging Decision Tree 




from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
# max_samples: maximum size 0.5=50% of each sample taken from the full dataset
# max_features: maximum of features 1=100% taken here all 10K 
# n_estimators: number of decision trees 
bg=BaggingClassifier(DecisionTreeClassifier(max_depth=2),max_samples=10,max_features=10,n_estimators=10)
bg.fit(x_train, y_train)





bg_base_accuracy_train =  bg.score(x_train, y_train)
bg_base_accuracy_test =  bg.score(x_test, y_test)


print("Training Accuracy: %.2f%%" % (bg_base_accuracy_train*100.0))
print("Testing Accuracy: %.2f%%" % (bg_base_accuracy_test*100.0))




# ## 3.5 Random Forest Model




rf = RandomForestClassifier(n_estimators=1)
rf.fit(x_train,y_train)





rf_base_accuracy_train =  rf.score(x_train, y_train)
rf_base_accuracy_test =  rf.score(x_test, y_test)

print("Training Accuracy: %.2f%%" % (rf_base_accuracy_train*100.0))
print("Testing Accuracy: %.2f%%" % (rf_base_accuracy_test*100.0))





# ## 3.6 Support Vector Machine




from sklearn.svm import SVC

lsvc = SVC(C=0.0001, class_weight=None, max_iter=10000,
          verbose=0)

lsvc.fit(x_train, y_train)

svc_base_accuracy_train =  lsvc.score(x_train, y_train)
svc_base_accuracy_test =  lsvc.score(x_test, y_test)


print("Training Accuracy: %.2f%%" % (svc_base_accuracy_train*100.0))
print("Testing Accuracy: %.2f%%" % (svc_base_accuracy_test*100.0))



# ## 3.7 KNN




from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)


knn_base_accuracy_train =  knn.score(x_train, y_train)
knn_base_accuracy_test =  knn.score(x_test, y_test)

print("Training Accuracy: %.2f%%" % (knn_base_accuracy_train*100.0))
print("Testing Accuracy: %.2f%%" % (knn_base_accuracy_test*100.0))


# # 3.8 Naive Bayes



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)


gnb_base_accuracy_train =  gnb.score(x_train, y_train)
gnb_base_accuracy_test =  gnb.score(x_test, y_test)

print("Training Accuracy: %.2f%%" % (gnb_base_accuracy_train*100.0))
print("Testing Accuracy: %.2f%%" % (gnb_base_accuracy_test*100.0))





models = ['LR', 'DT','ADB', 'RF', 'SVM', 'KNN', 'GNB']
test_accuracy = [lg_base_accuracy_test, dt_base_accuracy_test,adb_base_accuracy_test, rf_base_accuracy_test, svc_base_accuracy_test,knn_base_accuracy_test,gnb_base_accuracy_test]
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(models, test_accuracy, color ='maroon',
        width = 0.4)
 
plt.xlabel("base models")
plt.ylabel("test accuracy")
plt.title("comparison of the different base models")
plt.show()


# # 4. Saving the best model


# Saving a model
dump(adb, 'adaboost.dump')




