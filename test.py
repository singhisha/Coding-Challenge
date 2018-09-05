#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 14:54:30 2018

@author: ishaMac
"""

import challenge
import pandas as pd
[classifier, encoder] = challenge.my_classifier()
[classifier, encoder] = challenge.load_classifier()

# pre process the test dataset
test_dataset = pd.read_csv("test_data.csv")
test_dataset['launched'] = pd.to_datetime(test_dataset['launched'])
test_dataset['deadline'] = pd.to_datetime(test_dataset['deadline'])
days = (test_dataset['deadline'] - test_dataset['launched'])
test_dataset['days'] = days.astype('timedelta64[D]')
X_test = test_dataset.iloc[:, [2, 3, 4, 6, 8, 9, 11, 12, 13, 14]].values
X_test = encoder.transform(X_test)
    
# Predicting the Test set results
y_test_pred = classifier.predict(X_test)
y_test_pred = (y_test_pred > 0.5)
