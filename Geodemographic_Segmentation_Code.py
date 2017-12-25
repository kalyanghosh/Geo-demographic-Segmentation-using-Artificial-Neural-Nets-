# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:05:02 2017

@author: Kalyan
"""

#Creating a Geodemographic Segmentation of customers of bank:

#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Importing the keras DL Library
import keras
#Importing the Sequential library from keras
from keras.models import Sequential 
#Importing the Dense library from keras
from keras.layers import Dense 

class Geodemographic_Segmentation(object):
    
    #Declaring the member variables of the class 
    #which are accessible to all the methods of the class
    dataset=None
    X=None
    y=None
    X_train=None
    X_test=None
    y_train=None
    y_test=None
    classifier=None
    y_pred=None
    
    #Constructor method
    
    def __init__(self,dataset):
        self.dataset=dataset
        
    #Data preprocessing method
        
    def data_preprocessing(self,dataset):
        X=dataset.iloc[:,3:13].values
        y=dataset.iloc[:,13].values
    
        #Encoding the categorical features in the dataset
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        labelencoder_X_1 = LabelEncoder()
        X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
        labelencoder_X_2 = LabelEncoder()
        X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
        onehotencoder = OneHotEncoder(categorical_features = [1])
        X = onehotencoder.fit_transform(X).toarray()
        X = X[:, 1:]
        
        return X,y
    
    #Dividing the data into Training and Test dataset
    
    def train_test_split(self,X,y):
        
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
        
        return X_train,X_test,y_train,y_test
        
    #Performing Feature Scaling
    
    def feature_scaling(self,X_train,X_test):
        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler()
        X_train=sc.fit_transform(X_train)
        X_test=sc.fit_transform(X_test)
        
        return X_train,X_test
    
    
    #Defining and creating the Arificial Neural Network using Keras
    
    def create_ANN(self,X_train,y_train):
        
        # Initialising the ANN
        classifier = Sequential()

        # Adding the input layer and the first hidden layer
        classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

        # Adding the second hidden layer
        classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

        # Adding the output layer
        classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

        # Compiling the ANN
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        # Fitting the ANN to the Training set
        classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
        
        return classifier
    
    #Fitting the learned classifier on the Test data to make predictions and compute the 
    #performance statitics of the ANN
    
    def test_classifier(self,classifier,X_test,y_test):
        
        y_pred=classifier.predict(X_test)
        y_pred=(y_pred>0.5)
        
        from sklearn.metrics import confusion_matrix
        cm=confusion_matrix(y_test,y_pred)
        print (cm)
        TP=cm[0][0]
        TN=cm[1][1]
        FP=cm[0][1]
        FN=cm[1][0]
        Accuracy=(TP+TN)/(TP+TN+FP+FN)
        Precison=TP/(TP+FP)
        Recall=TP/(TP+FN)
        F1score=(2*Precison*Recall)/(Precison+Recall)
        print ("******Model successfully tested******")
        print ("******Model Statistics ******")
        print ("Accuracy of Model = ",Accuracy)
        print ("Precison of Model = ",Precison)
        print ("Recall of Model   = ",Recall)
        print ("F1score of Model  = ",F1score)
        print ("****************************")
        
    
    
#Importing the dataset    
dataset=pd.read_csv('Churn_Modelling.csv')

#Creating an instance of the class
        
gds=Geodemographic_Segmentation(dataset)

#Invoking the data_preprocessing method
X,y=gds.data_preprocessing(dataset)

#Invoking train_test_split method 
X_train,X_test,y_train,y_test=gds.train_test_split(X,y)

#Invoking the feature_scaling method
X_train,X_test=gds.feature_scaling(X_train,X_test)

#Creating the ANN and getting the learned classifier
classifier=gds.create_ANN(X_train,y_train)

#Fitting the learned classifier on the test data
gds.test_classifier(classifier,X_test,y_test)

    

