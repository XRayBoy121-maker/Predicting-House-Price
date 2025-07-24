#Importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import _california_housing
housing=_california_housing.fetch_california_housing()
#creating a DataFrame from the dataset
df=pd.DataFrame(columns=housing.feature_names, data=housing.data)
df['Price'] = housing.target
#For reducing the R and Adjusted R values, we will drop the features that are not significant
df.drop(columns=['AveBedrms','Latitude'] , inplace=True)
#Statistical summary of the DataFrame
# print(df.describe())
#Exploatry Data Analysis
#Finding the correlation between features and the target variable
# print(df.corr())
#Plotting the pairplot to visualize relationships between features
# sns.pairplot(df)
#Displaying the outliers using boxplot
fig,ax=plt.subplots(figsize=(15,15))
# sns.boxplot(data=df, ax=ax)
plt.savefig('Not Normalized boxplot.jpg')
plt.boxplot(df)
#Normalizing the data and splitting it into training and testing sets
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
# print(x , "\n", y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
# print("Training set size:", x_train," \n", y_train)
# print("Testing set size:", x_test, " \n", y_test)
from sklearn.preprocessing import StandardScaler
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_train_norm = scaler_x.fit_transform(x_train)  
x_test_norm = scaler_x.transform(x_test)   
y_train_norm = scaler_y.fit_transform(y_train)  
y_test_norm = scaler_y.transform(y_test)        
fig, ax = plt.subplots(figsize=(15, 15))
# Convert normalized features back t
x_train_norm= pd.DataFrame(x_train_norm, columns=x_train.columns)
# sns.boxplot(data=x_train_norm, ax=ax)
plt.savefig('Normalized_boxplot.jpg')
# plt.show()
# print(y_train_norm, "\n", y_test_norm) 
x_train_norm = pd.DataFrame(scaler_x.fit_transform(x_train),columns=x_train.columns)
x_test_norm = pd.DataFrame(scaler_x.transform(x_test),columns=x_test.columns)
# 4. Scale target and convert back to DataFrames
y_train_norm = pd.DataFrame(scaler_y.fit_transform(y_train),columns=['Price'])  # Use your target column name
y_test_norm = pd.DataFrame(scaler_y.transform(y_test),columns=['Price'])
# Training the model using Linear Regression
from sklearn.linear_model import LinearRegression
regression= LinearRegression()
regression.fit(x_train_norm, y_train_norm)
# print("Coefficients:", regression.coef_)
# print("Intercept:", regression.intercept_)

# Making predictions on the test set
x_pred = regression.predict(x_test_norm)
print("Predictions:", x_pred)
# Evaluating the model
from sklearn.metrics import mean_squared_error, r2_score
residual = y_test_norm - x_pred

#residual plot

# sns.displot(residual, kind='kde')
# plt.title('Residual Plot')
# plt.savefig('Residual_Plot.jpg')
# plt.show()

#Performance metrics
mse = mean_squared_error(y_test_norm, x_pred)
r2 = r2_score(y_test_norm, x_pred)
mae=mean_squared_error(y_test_norm, x_pred)
print("Mean Squared Error:", mse)   
print("R-squared:", r2)
print("Adjusted R-squared:", 1 - (1 - r2) * (len(y_test_norm) - 1) / (len(y_test_norm) - x_test_norm.shape[1] - 1))
print("Mean Absolute Error:", mae)