"""Parkinson's diseases Detection

Importing the Dependencies
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

"""Data Collection & Analysis"""

# loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('C:\\Users\\BS COMP\\Desktop\\python\\parkinsons.csv')

# printing the first 5 rows of the dataframe
parkinsons_data.head()

# number of rows and columns in the dataframe
parkinsons_data.shape

# getting more information about the dataset
parkinsons_data.info()

# checking for missing values in each column
parkinsons_data.isnull().sum()

# getting some statistical measures about the data
parkinsons_data.describe()

# distribution of target Variable
parkinsons_data['status'].value_counts()

"""
1 --> Parkinson's Positive

0 --> Healthy
"""

# grouping the data bas3ed on the target variable
parkinsons_data.groupby('status').mean()

"""Data Pre-Processing

Separating the features & Target
"""

X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']

print(X)

print(Y)

"""Splitting the data to training data & Test data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Data Standardization"""

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

print(X_train)

"""Model Training

Support Vector Machine Model
"""

model = svm.SVC(kernel='linear')

# training the SVM model with training data
model.fit(X_train, Y_train)

"""Model Evaluation

Accuracy Score

"""

# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy score of training data : ', training_data_accuracy)

# accuracy score on testing data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy score of test data : ', test_data_accuracy)

"""Building a Predictive System

"""

input_data = (124.44500,135.06900,117.49500,0.00431,0.00003,0.00141,0.00167,0.00422,0.02184,0.19700,0.01241,0.01024,0.01685,0.03724,0.00479,25.13500,0.553134,0.775933,-6.650471,0.254498,1.840198,0.103561)
# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)


if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons")