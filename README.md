# Geo-demographic-Segmentation-using-Artificial-Neural-Nets-
This repository contains codebase and datafiles for the project titled "Geo demographic Segmentation using Artificial Neural Networks "

# The objective of this project is to create a Geo-Demographic segmentation model for a bank using Artificial Neural Networks.

# The dataset for this project is the Churn_Modelling.csv dataset which contains the data about 100,000 customers of the bank

#The steps followed for this project is as below:

1. The "Keras" Deep Learning library is used for this project

2. class Geodemographic_Segmentation:

   This class is the template of the code. This class contains methods related to the project
   
2.1 def __init__(self,dataset):

    This is the constructor method
   
2.2 def data_preprocessing(self,dataset):

    This method is responsible for performing the basic data preprocessing
    
2.3 def train_test_split(self,X,y):

    This method is responsible for dividing the dataset into Test and Train
    
2.4 def feature_scaling(self,X_train,X_test):

    This method is used for feature scaling 

2.5 def create_ANN(self,X_train,y_train):

    This method builds the 3 layer Artificial Neural Network using the Keras library
    
 
2.6 def test_classifier(self,classifier,X_test,y_test):

    This method fits the learned classifier on the training dataset to the test data and calculates the prediction statistics
    
NOTE : The model was trained on a Linux EC2 instance of Amazon Web Service. The model took 3 minutes to training and prediction 
       using 100 epochs
       
       The statistics of the project are as follows:
       
       1. Cross Entropy Loss of the Model = 0.3402
       
       2. Accuracy of Classification = 85.96%
       
