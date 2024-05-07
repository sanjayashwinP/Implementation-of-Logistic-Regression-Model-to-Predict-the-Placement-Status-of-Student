# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.   
2.Print the placement data and salary data.    
3.Find the null and duplicate values.  
4.Using logistic regression find the predicted values of accuracy , confusion matrices.    
5.Display the results
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
### Opening File:
![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/2ad40d29-1252-46dd-afcc-0b248b89f418)

### Droping File:
![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/fa1f63ff-cee7-41bc-a45b-09fa16a968d0)

### Duplicated():
![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/3151cfa1-22e1-4582-8d2d-c1348ee755b5)

### Label Encoding:
![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/58f9d71a-ee8b-4a7b-b6ce-4f1a49837751)

### Spliting x,y:
![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/7ada83e6-9e0a-4b07-a799-46bd8dfc2704)
![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/986283f3-28c7-4b2d-bea6-aee10bdfedbe)

### Prediction Score
![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/bd32fa36-1d2a-441d-8f6d-5185846f1cb9)

### Testing accuracy
![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/904ce3b8-975a-467f-91b0-eca42bcd451d)

### Classification Report
![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/899227a0-9539-42d0-bdf1-2edef822cbd3)

### Testing Model:
![image](https://github.com/sanjayashwinP/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/147473265/427b0181-5312-4cf5-84b7-dbbe07d07195)

## RESULT:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
