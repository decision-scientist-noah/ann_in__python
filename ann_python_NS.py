# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 19:30:16 2020

@author: nsheldon
"""

import os
os.chdir('C:\Users\nsheldon\Downloads\Deep_Learning_A_Z\Volume 1 - Supervised Deep Learning\Part 1 - Artificial Neural Networks (ANN)\Section 4 - Building an ANN\Artificial_Neural_Networks')

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Python script for confusion matrix creation. 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1]
y = dataset.iloc[:, -1]

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder

labelencoder_X_gender = LabelEncoder()
X.iloc[:, 2] = labelencoder_X_gender.fit_transform(X.iloc[:, 2])
X = pd.concat([X, pd.get_dummies(X['Geography'])], axis=1)
X.drop(['Spain','Geography'],axis = 1, inplace =True)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, init = 'uniform' , activation = 'relu' , input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

#Adding the output layer
classifier.add(Dense(units = 1 , kernel_initializer = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, epochs = 100, batch_size = 10)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy
accuracy_score(y_test, y_pred) 

#Classification Report
classification_report(y_test, y_pred)
# =============================================================================
# 
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $60000
# Number of Products: 2
# Does this customer have a credit card ? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $50000
# =============================================================================

new_pred = np.array([[600,1 , 40, 3, 60000, 2, 1, 1, 50000, 1,0 ]])

new_prediction = classifier.predict(sc.transform(new_pred))
new_pred = new_prediction > 0.5






























