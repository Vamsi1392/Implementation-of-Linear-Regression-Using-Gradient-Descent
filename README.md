
# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1. Load necessary libraries like numpy, pandas, and StandardScaler for scaling features and target variables.

2.Implement a linear_regression function that uses gradient descent to minimize the cost function and compute optimal model parameters (theta).

3.Read the dataset from a CSV file, extract features (X) and target variable (y), and convert them to numeric types.

4.Scale both the feature matrix (X1) and target variable (y) using StandardScaler to improve gradient descent performance.

5.Call the linear_regression function with the scaled features and target to compute the model parameters (theta).

6.Scale new data using the same scaler, apply the model parameters (theta), and inverse scale the prediction to get the final result.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: vamsidhar reddy
RegisterNumber:  212224040205

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions - y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_scaled=scaler.fit_transform(X1)
Y1_scaled=scaler.fit_transform(y)
print(X1_scaled)
print(Y1_scaled)
theta=linear_regression(X1_scaled,Y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")



*/
```

## Output:
![Screenshot 2025-03-07 183700](https://github.com/user-attachments/assets/7c7cd804-88a5-45ac-9a85-997b894a402c)


![Screenshot 2025-03-07 183748](https://github.com/user-attachments/assets/695cd6ec-28b5-4ba9-aa6a-6c1fa7d4367d)



![Screenshot 2025-03-07 183832](https://github.com/user-attachments/assets/87d929d9-6130-4027-8974-21661ed7435c)


![Screenshot 2025-03-07 183918](https://github.com/user-attachments/assets/49957d31-30df-48dc-ad32-d44125b56e2f)


![Screenshot 2025-03-07 183948](https://github.com/user-attachments/assets/3f962597-0ce8-4d0f-97d8-0d87062c869f)







## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
