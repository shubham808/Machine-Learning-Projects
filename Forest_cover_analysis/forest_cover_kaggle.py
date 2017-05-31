
import sys
import csv
import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
import pandas
from sklearn.feature_selection import SelectKBest, f_classif 

from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

train=pandas.read_csv("foresT_train.csv")
test=pandas.read_csv("foresT_test.csv")

train.drop('Id',axis=1,inplace=True)
ids=test['Id']
test.drop('Id',axis=1,inplace=True)
labels_train=train['Cover_Type']
train.drop('Cover_Type',axis=1,inplace=True)

labels_train = np_utils.to_categorical(labels_train)

to_scale=['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology', 
             'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
             'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
             'Horizontal_Distance_To_Fire_Points']
    


for x in to_scale:
    scaler = StandardScaler().fit(train[x].values.astype('float_'))
    train[x] = scaler.transform(train[x].values.astype('float_'))
    test[x] = scaler.transform(test[x].values.astype('float_'))

model = Sequential()
model.add(Dense(512, input_dim=54))
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(np.array(train),labels_train,validation_split=0.1,epochs=70)
out = model.predict_classes(np.array(test))


idk = open('foresT_test.csv','r')
idk = csv.DictReader(idk)
pred = []
for row in idk:
    pred.append(row["Id"])
est = csv.writer(open('result_forest.csv', 'w'), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
est.writerows([["Id","Cover_Type"]])
k=0
for x in pred:
    est.writerows([[str(x), str(out[k])]])
    k+=1

