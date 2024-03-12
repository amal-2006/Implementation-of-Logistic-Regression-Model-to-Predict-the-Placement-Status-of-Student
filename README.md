# Implementation of Logistic Regression Model to Predict the Placement Status of Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.

2. Print the placement data and salary data.

3. Find the null and duplicate values.

4. Using logistic regression find the predicted values of accuracy , confusion matrices.

5. Display the results.


## Program:

```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: AMALJOSH MAADHAV J
RegisterNumber:  212223230012

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='liblinear')
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:

### TOP 5 ELEMENTS
![Screenshot 2024-03-12 092235](https://github.com/amal-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/148410730/4b9a5539-5a71-4069-ab35-a810550d06ac)

![Screenshot 2024-03-12 092242](https://github.com/amal-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/148410730/a54430e9-e9e1-45c6-a6c3-288d02355050)

![Screenshot 2024-03-12 092252](https://github.com/amal-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/148410730/5b8eb7e7-79d8-43ec-938c-322844a523eb)
### DATA DUPLICATE
![Screenshot 2024-03-12 092257](https://github.com/amal-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/148410730/195bc4ec-ad03-4b51-a57b-bc200c43325a)

### PRINT DATA
![Screenshot 2024-03-12 092303](https://github.com/amal-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/148410730/1f8f6994-11bc-42b9-8e76-d0b174fdbcf5)

### DATA_STATUS
![Screenshot 2024-03-12 092354](https://github.com/amal-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/148410730/942198b9-6ddb-44a4-abc8-9c5c11a5e9f0)

### Y_PREDICTION ARRAY
![Screenshot 2024-03-12 092359](https://github.com/amal-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/148410730/b0874ebf-c4e9-4c0b-84f3-a8b4a00f521d)

### ACCURACY VALUE
![Screenshot 2024-03-12 092511](https://github.com/amal-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/148410730/77246ff1-713d-430f-b61a-a2c15ae1fef9)

### CONFUSION ARRAY
![Screenshot 2024-03-12 092515](https://github.com/amal-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/148410730/a00c9ad8-5c04-4cd2-b7ab-041cf83a07b6)

### CLASSIFICATION REPORT
![Screenshot 2024-03-12 092519](https://github.com/amal-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/148410730/12ee591c-cc9d-4e00-92ef-0f47fb6c057c)

### PREDICTION
![Screenshot 2024-03-12 092524](https://github.com/amal-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/148410730/feb9a481-bfe6-4859-8f03-a4c0c582f4b1)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
