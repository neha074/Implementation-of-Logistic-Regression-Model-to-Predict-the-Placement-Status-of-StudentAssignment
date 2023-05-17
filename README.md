# Ex 04 Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.import pandas module.

2.Read the required csv file using pandas . 3.Import LabEncoder module.

4.From sklearn import logistic regression.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.print the required values.

8.End the program.

## Program:
~~~
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: V Jaivignesh
RegisterNumber: 212220040055


import pandas as pd

data = pd.read_csv("Placement_Data.csv")

print(data.head())


data1 = data.copy()
data1= data1.drop(["sl_no","salary"],axis=1)
print(data1.head())

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()

data1["gender"] = lc.fit_transform(data1["gender"])
data1["ssc_b"] = lc.fit_transform(data1["ssc_b"])
data1["hsc_b"] = lc.fit_transform(data1["hsc_b"])
data1["hsc_s"] = lc.fit_transform(data1["hsc_s"])
data1["degree_t"]=lc.fit_transform(data["degree_t"])
data1["workex"] = lc.fit_transform(data1["workex"])
data1["specialisation"] = lc.fit_transform(data1["specialisation"])
data1["status"]=lc.fit_transform(data1["status"])

print(data1)

y = data1["status"]

x = data1.iloc[:,:-1]
print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
print(lr.fit(x_train,y_train))


y_pred = lr.predict(x_test)
print(y_pred)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)


from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)


from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

#for prediction lets take the first value from data 1

prediction = [1,67,1,91,1,1,58,2,0,55,1,58.80]
print(lr.predict([prediction])) # status should be 1 

#now we predict for random value asuuming gender ssc_p ssc_b .... be

prediction = [1,80,1,90,1,1,90,1,0,85,1,85]

print(lr.predict([prediction]))

~~~~
## Output:
1.Placement data
<img width="1012" alt="Screenshot 2023-05-17 at 2 14 21 PM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/10605748-e593-488f-a0ab-79607fb3e659">

2.Salary data
<img width="1012" alt="Screenshot 2023-05-17 at 2 14 27 PM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/8893e4e2-94c2-4b29-b649-e06b51d5cfb5">

3.Checking the null() function


<img width="413" alt="Screenshot 2023-05-17 at 2 14 34 PM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/a243870c-bdb8-494a-9847-7fa2d1ac75c6">

4. Data Duplicate

<img width="816" alt="Screenshot 2023-05-17 at 2 14 42 PM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/06a3754b-e4b2-46e0-9542-46c783976db0">


<img width="588" alt="Screenshot 2023-05-17 at 2 14 47 PM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/7e9bdf54-bebd-4193-b4f5-26e9c6f6894b">

5. Print data

<img width="790" alt="Screenshot 2023-05-17 at 2 14 54 PM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/f1545105-0c46-4f70-9263-00ad91c5f627">

6. Data-status


<img width="431" alt="Screenshot 2023-05-17 at 2 15 01 PM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/3ebfb6cb-a04a-4f1a-8265-ea8f8325b100">


8. y_prediction array

<img width="682" alt="Screenshot 2023-05-17 at 2 15 09 PM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/834b6d00-6fdb-43de-9acd-4394d9f2bfcc">



8.Accuracy value

<img width="419" alt="Screenshot 2023-05-17 at 2 15 17 PM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/26e3e039-f623-43c1-b444-d133f2c31b76">



9. Confusion array



<img width="232" alt="Screenshot 2023-05-17 at 2 15 22 PM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/1cb1e74e-3c01-4f5c-a060-cb6d27bccbda">

10. Classification report


<img width="529" alt="Screenshot 2023-05-17 at 2 15 36 PM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/96676466-6396-4bbb-add7-6b8d83a81bb5">

11.Prediction of LR


<img width="217" alt="Screenshot 2023-05-17 at 2 15 47 PM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/66fb3ffd-bbb9-4265-b5ea-da309c9a6e77">


<img width="217" alt="Screenshot 2023-05-17 at 2 15 50 PM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/42f52ddf-1aed-43d0-8ee9-af4ecf3c0810">

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
