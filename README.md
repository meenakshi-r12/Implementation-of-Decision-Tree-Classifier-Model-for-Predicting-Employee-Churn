# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Required Libraries.
2. Use chardet to detect the encoding of the file spam.csv.
3. Load the CSV file using the encoding (in this case, 'windows-1252').
4. Display the first few rows and general information.
5. Ensure the dataset does not contain null or missing values.
6. Feature (x) = text messages (column "v2").
7. Label (y) = spam/ham classification (column "v1").
8. Split the Data into Training and Testing Sets.
9. Use CountVectorizer to convert text data into numerical vectors.
10. Use Support Vector Classifier to train on the vectorized data.
11. Predict whether messages are spam or not.
12. Use accuracy score to evaluate the model performance

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: MEENAKSHI.R
RegisterNumber: 212224220062
*/
```
```
import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()

data.info()
data.isnull().sum()
data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:

![Screenshot (125)](https://github.com/user-attachments/assets/2a20e988-ecfc-404e-b318-8a5363791d5b)

![Screenshot (126)](https://github.com/user-attachments/assets/577678e6-1ccf-4a29-8e78-5d71f6833e06)

![Screenshot (127)](https://github.com/user-attachments/assets/1f5054b7-cf09-4fdf-a77d-71edb47e6797)

![Screenshot (128)](https://github.com/user-attachments/assets/cc2aa094-38e3-4aba-8675-b7b4216efe49)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
