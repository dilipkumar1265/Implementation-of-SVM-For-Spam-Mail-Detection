# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. Import the python pandas library as pd.
3. Read the contents of the Spam csv file.
4. Display the first 5 rows of the dataset using head().
5. Assign x as v1 values and y as v2 values.
6. From sklearn library select the feature extraction and import CountVectorizer.
7. CountVectorizer will convert the Text to Numerical Data.
8. From sklearn library import Support Vector Classifier (ie. SVC).
9. Predict the x_test using SVC.
10. Print the accuracy of the SVM Model.
11. Stop the program 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: DILIP KUMAR R
RegisterNumber:  212222040037
*/
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["EmailText"].values
y=data["Label"].values
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["EmailText"].values
y=data["Label"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
## data.head()
![Screenshot 2023-11-12 150256](https://github.com/dilipkumar1265/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119065291/296e0daa-947e-4e89-81be-ca58cfee527e)
## data.info()
![Screenshot 2023-11-12 150303](https://github.com/dilipkumar1265/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119065291/5afd8b15-7091-40df-a272-33ba737a198d)
## data isnull()& sum()
![Screenshot 2023-11-12 150308](https://github.com/dilipkumar1265/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119065291/21ed5412-dfd0-425c-a9bc-70f814bc4bc6)
## Y_prediction()
![Screenshot 2023-11-12 150314](https://github.com/dilipkumar1265/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119065291/d9779ca4-a319-475a-82d7-aceebc68abb4)
## Accuracy:
![Screenshot 2023-11-12 150336](https://github.com/dilipkumar1265/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119065291/364a62eb-934b-424f-806a-686346908cbe)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
