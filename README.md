# Machine Learning Pipeline
## *Attiriton Rate Prediction*
Name : Lau Sook Han Gayle
Date: 31/10/22
Email: laugayle@gmail.com


## Data Overview
### Independent Features

Attribute Description
*`Age` - Age of the member when signing up as a member
*`Gender` - Gender of the member 
*`Monthly Income` - Monthly declared income of the member in SGD
*`Travel Time` - Estimated amount of time needed to travel to the club from home (mins)
*`Qualification` - Education qualification level of the member (1-Diploma, 2-Bachelors, 3-Masters, 4-PH.D)
*`Work Domain` - Work domain of the member
*`Usage rate` - Average number of days in a week visiting the country club
*`Branch` - Location of the branch that the member visits
*`Membership` - Membership tier (1-Normal 2-Bronze 3-Silver 4-Gold)
*`Months` - Number of months as a member of the country club
*`Birth Year` - Year the member was born
*`Usage Time` - Average number of hours spent in the country club per visit
*`Usage` - Usage Time * Usage Rate (new)


### Target Feature
* `Target Feature`: Attrition (If the member left: 0 = No, 1 = Yes)

### Unique Identifying Features
* `Member Unique ID`:  Unique member ID (removed)

### Catergorical Features
* `Categorical features`: `Qualification`, `Work Domain`, `Branch`, `Membership`

### Binary Features
* `Binary features`: `Gender`, `Attrition`

### Numerical Features
* `Numerical Features`: `Age`,`Travel Time`,`Monthly Income`, `Usage Rate`, `Months`, `Birth Year`, `Usage Time`

# Sypnopsis of the problem. 
* **Classification**: predict member attrition using the provided dataset to help a country club to
formulate policies to reduce attrition. In your submission, you are to evaluate at least 3 suitable models
for predicting member attrition.

## Overview of Submitted folder
.
├── eda.ipnyb
├── data
│   └── score.db # removed
├── requirements.txt
├── run.sh
└── src
    ├──
    └── run.py

## Executing the pipeline
**run.py**
Stepts:
**1. Imports the data from .db file
**2. Data Cleaning (data is cleaned - 'Age', 'Monthly Income', `Birth Year`, 'Qualification', 'Travel Time'
**3. Feature Engineering (one-hot encoding/ordinal encoding for categorical data)
**4. Data split into training and testing data (80/20)
**5. Building of the Models

*Logistic Regression
*Decision Tree Model
*Boosting Decision Tree
*Bagging Decision Tree
*Random Forest Model
*Support Vector Machine
*KNN
*Naive Bayes

**6. Results**

Accuracy and Recall


## Running of machine learning pipeline.
Machine learning model created with python 3 and bash script.

### Installing dependencies
Paste the following command on your bash terminal to download dependencies
```sh
pip install -r requirements.txt
```


### Running the Machine Learning Pipeline
Past the followin command on your bash terminal to grant permission to execute the 'run.sh' file
```sh
chmod +x run.sh
```
Paste the following command on the bash terminal to run the machine learning programme
```sh
./run.sh
```







