#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 11:44:38 2017
##GOT 0.98214 on KAGGLE
@author: shubham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test = test.values
data = data.values
print(data.shape)

features = data[:, 1:]
labels = data[:, :1]

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25,
                                                                               random_state=42)

features_train = features_train.reshape(features_train.shape[0],28,28,1)
features_test = features_test.reshape(features_test.shape[0],28,28,1)

labels_train = np_utils.to_categorical(labels_train,10)
labels_test = np_utils.to_categorical(labels_test,10)

#define our model
model=Sequential()
#declare input layer use tensorflow backend meaning depth comes at the end
model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
print(model.output_shape)

model.add(Convolution2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#conver to 1D by flatten
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

#compile model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(features_train,labels_train,validation_split=0.1)
score = model.evaluate(features_test,labels_test,verbose=0)
print(score)
test = test.reshape(test.shape[0],28,28,1)

pred = model.predict_classes(test,verbose=0)

pred = pd.DataFrame(pred)

pred['ImageId'] = pred.index + 1
pred = pred[['ImageId', 0]]
pred.columns = ['ImageId', 'Label']

pred.to_csv('pred.csv', index=False)
print("Done!")