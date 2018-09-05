#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 09:45:00 2018

@author: ishaMac
"""

def my_classifier():
    import numpy as np
    import pandas as pd
    
    # importing the dataset
    dataset = pd.read_csv("data.csv")
    
    # removing rows with NaN values
    dataset.isnull().any()
    dataset = dataset.dropna()
    dataset.isnull().any()
    
    dataset['launched'] = pd.to_datetime(dataset['launched'])
    dataset['deadline'] = pd.to_datetime(dataset['deadline'])
    days = (dataset['deadline'] - dataset['launched'])
    dataset['days'] = days.astype('timedelta64[D]')
    X = dataset.iloc[:, [2, 3, 4, 6, 8, 10,12, 13, 14, 15]].values
    y = dataset.iloc[:, 9].values
    
    # data preprocessing
    # Encoding the Dependent Variable
    y[y == 'successful'] = 1
    y[y != 1] = 0
    y = y.astype(float)
    
    # Encoding categorical data
    # likelihood encoding/impact coding/target coding
    import category_encoders as ce
    encoder = ce.TargetEncoder(X)
    encoder.fit(X, y)
    X = encoder.transform(X)
    
    # Importing the Keras libraries and packages
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    #Dense function is used to add a fully connected layer in ANN
    #We won't need to specify input_dim in the next layers as they will already know what to expect
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
    classifier.add(Dropout(p = 0.1))
        
    # Adding the second hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
    
    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Fitting the ANN to the Training set
    classifier.fit(X, y, batch_size = 10, epochs = 20)
    
    import pickle
    classifier.save('model.pkl','wb')
    pickle.dump(encoder, open('encoder.pkl', 'wb'))
    
    return classifier, encoder

def load_classifier():
    import pickle
    from keras.models import load_model
    classifier = load_model('model.pkl')
    encoder = pickle.load(open('encoder.pkl', 'rb'))