# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## Aim:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe
4. Plot the required graph both for test data and training data.
5. .Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Yuvadarshini S
RegisterNumber:  212221230126

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
dh.head()

df.tail()

#segregating data to variables
x = df.iloc[:,:-1].values
x

y = df.iloc[:,1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size = 1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual value
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="magenta")
plt.plot(x_train,regressor.predict(x_train),color="black")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data 
plt.scatter(x_test,y_test,color="orange") 
plt.plot(x_test,regressor.predict(x_test),color="gray") 
plt.title("Hours vs Scores(Training Set)") 
plt.xlabel("Hours") 
plt.ylabel("Scores") 
plt.show()

mse = mean_squared_error(y_test,y_pred) 
print('MSE = ',mse) 

mae = mean_absolute_error(y_test,y_pred) 
print('MAE = ',mae) 

rmse = np.sqrt(mse) 
print('RMSE',rmse)

*/
```

## Output:
### df.head()
![s1](https://user-images.githubusercontent.com/93482485/228900151-c14c3a85-8fdf-4e71-8373-dd1a993cbe15.jpg)

### df.tail()
![s2](https://user-images.githubusercontent.com/93482485/228900278-2d8fe404-e60c-4052-8296-8252e8009617.jpg)

### Array value of x
![s3](https://user-images.githubusercontent.com/93482485/228900295-2ff58552-36cf-4d40-9d44-62eb42f5dce6.jpg)

### Array value of y
![s4](https://user-images.githubusercontent.com/93482485/228900323-37c9506b-6c1d-460e-afa4-20149a6b32fa.jpg)

### Value of y prediction
![s5](https://user-images.githubusercontent.com/93482485/228900355-797a71a6-9abf-48f6-8ddb-d4b87546ece2.jpg)

### Array values of y test
![s6](https://user-images.githubusercontent.com/93482485/228900383-9f7eab1e-d983-4172-bcfe-fb7122b691c4.jpg)

### Training Set Graph
![s7](https://user-images.githubusercontent.com/93482485/228900423-f959fe5f-9417-4674-bdc7-8900aac9753c.jpg)

### Test Set Graph
![s8](https://user-images.githubusercontent.com/93482485/228900458-41836277-ba92-4e86-8d77-01f95d5b7c1c.jpg)

### Values of MSE, MAE and RMSE
![s9](https://user-images.githubusercontent.com/93482485/228900486-25d7de5b-25d8-4ca7-a4ed-5633db0a5b75.jpg)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
