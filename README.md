# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SANJAY ASHWIN P.
RegisterNumber: 212223040181. 
*/

```
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/43114fa8-03cf-430f-82cb-a4176b3e929b)

![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/aa042a8b-c886-427d-9154-8fbd85229e43)

![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/f5cf572c-1ffd-4f07-b9da-faf26e8c3277)

![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/49c5f707-9220-49cb-bdea-e0612d38d93a)

![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/81b97d9b-9c5f-4dfe-964a-6538816d414c)

![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/2d48a8ce-a439-4b1a-a851-094c9f365fcf)

![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/995954e4-d166-4636-b57c-27f41a93538c)

![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/ed2d2ed0-2fa1-4639-99e3-48a551694f57)

![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/dc5ff3ee-6c88-4b40-9729-89c786612682)

![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/c0f560a8-1b86-4e58-96ca-c263e1f496fc)

![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/67669994-02b0-493d-bcb7-fda9ae0ec62c)

![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/45d41c1d-146b-469d-a424-0453c98badc3)

![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/7663f995-a7e6-4ee8-b07c-614c8070f18f)

## RESULT:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
