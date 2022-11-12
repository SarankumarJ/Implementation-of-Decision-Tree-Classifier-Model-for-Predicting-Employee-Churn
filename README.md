# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```py
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Sarankumar J
RegisterNumber: 212221230087

#import packages
import pandas as pd
df=pd.read_csv("Employee.csv")
df.head()

#checking the data information and null presence
df.info()
df.isnull().sum()

#checking the count of left column
df["left"].value_counts()

#encoding categorical features to numeric
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()

#assigning x and y array and display them
x=df[["satisfaction_level","last_evaluation","number_project",
"average_montly_hours","time_spend_company","Work_accident",
"promotion_last_5years","salary"]]
x.head()
y=df["left"]
y

#splitting data into training and test
#implementing decision tree classifier in training model
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

#calcuating accuracy score
from sklearn import metrics
acc=metrics.accuracy_score(y_test,y_pred)
acc

#testing the model
dt.predict([[.5,.8,9,260,6,0,1,2]])
```

## Output:
![image](https://user-images.githubusercontent.com/94778101/201475248-371cf0fb-7b6b-4b64-a579-25d767cb5c60.png)

![image](https://user-images.githubusercontent.com/94778101/201475256-b1bc9b3d-3efa-4e23-81fe-8ad280ab1446.png)

![image](https://user-images.githubusercontent.com/94778101/201475259-814e43ee-42b3-4d29-8d2d-d66c3acda2a6.png)

![image](https://user-images.githubusercontent.com/94778101/201475266-814c91dc-deb6-4cf6-a780-c9213e9eee7c.png)

![image](https://user-images.githubusercontent.com/94778101/201475273-aeba450c-aa0e-4fc3-ba42-f397c8d62b71.png)

![image](https://user-images.githubusercontent.com/94778101/201475284-3255b4a4-f662-4473-92a7-daf5d6fc0ec2.png)

![image](https://user-images.githubusercontent.com/94778101/201475306-05b55341-8c2f-4e33-a59d-0c410e7145a5.png)

![image](https://user-images.githubusercontent.com/94778101/201475314-5c0da216-3072-4fcd-8795-7d3fea640216.png)

![image](https://user-images.githubusercontent.com/94778101/201475328-a93da560-789c-49c2-8779-670519e07bf0.png)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
