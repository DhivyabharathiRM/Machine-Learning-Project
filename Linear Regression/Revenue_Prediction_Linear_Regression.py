# Revenue Prediction using Simple Linear Regression

# Importing Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing Dataset
dataset = pd.read_csv('D:/Data Analytics/DA/Datasets/IceCreamData.csv')

# Analyzing the Data
print(dataset.head())
print(dataset.describe())
dataset.info()
print(dataset.isnull().sum())

# Visulazing the Data
sns.jointplot(x='Temperature', y='Revenue', data = dataset)
sns.pairplot(dataset)
sns.lmplot(x='Temperature', y='Revenue', data=dataset)

# Feature Selection
X=dataset.iloc[:,0].values
y=dataset.iloc[:,1].values

# Splitting the data into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Training the Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept =True)
regressor.fit(X_train.reshape(-1,1), y_train)

print('Linear Model Coefficient (m): ', regressor.coef_)
print('Linear Model Intercept (b): ', regressor.intercept_)

# Testing the Model
y_pred = regressor.predict(X_test.reshape(-1,1))

# Visualizing Training set result 
plt.scatter(X_train.reshape(-1,1), y_train, color = 'red')
plt.plot(X_train.reshape(-1,1), regressor.predict(X_train.reshape(-1,1)), color = 'blue')
plt.title('Temperature Vs Revenue (Training set)')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.show()

# Visualizing Testing set result 
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train.reshape(-1,1), regressor.predict(X_train.reshape(-1,1)), color = 'blue')
plt.title('Temperature Vs Revenue (Test set)')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.show()

# Actual vs Predicted
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(df)

plt.scatter(y_test,y_pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()
