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
Developed by: Neha.MA
RegisterNumber: 212220040100


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

<img width="1222" alt="Screenshot 2023-05-08 at 3 19 56 PM" src="https://user-images.githubusercontent.com/71516398/236794583-634639d4-b156-448f-bd85-b2c541ade2da.png">
<img width="1222" alt="Screenshot 2023-05-08 at 3 20 01 PM" src="https://user-images.githubusercontent.com/71516398/236794592-82e0b7a4-4f2e-41a3-9a37-6b695d69dc28.png">
<img width="475" alt="Screenshot 2023-05-08 at 3 20 10 PM" src="https://user-images.githubusercontent.com/71516398/236794601-152c7167-6f01-445f-9cc0-b7d447306aec.png">
<img width="864" alt="Screenshot 2023-05-08 at 3 20 30 PM" src="https://user-images.githubusercontent.com/71516398/236794604-77b04630-3999-4d9c-87ed-b9d56247f8b3.png">
<img width="823" alt="Screenshot 2023-05-08 at 3 20 49 PM" src="https://user-images.githubusercontent.com/71516398/236794611-279bfd50-dbb5-42f9-94ac-5c8342819da3.png">
<img width="850" alt="Screenshot 2023-05-08 at 3 21 03 PM" src="https://user-images.githubusercontent.com/71516398/236794614-91f39546-9dd9-467d-9d1d-05f314926d6b.png">
<img width="805" alt="Screenshot 2023-05-08 at 3 21 15 PM" src="https://user-images.githubusercontent.com/71516398/236794620-9a9e0615-b22f-4d2e-b331-7c557bec220e.png">
<img width="805" alt="Screenshot 2023-05-08 at 3 21 23 PM" src="https://user-images.githubusercontent.com/71516398/236794624-6fe6eb34-09db-4634-8ab3-e59f11fb750c.png">
<img width="805" alt="Screenshot 2023-05-08 at 3 21 30 PM" src="https://user-images.githubusercontent.com/71516398/236794626-673fbd0c-8240-4bcb-aeb0-66e46c34ade5.png">


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
