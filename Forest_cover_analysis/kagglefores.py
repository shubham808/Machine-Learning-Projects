import sys
import pandas as pd
import numpy as np
from keras.models import Sequential
 
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout


train=pandas.read_csv("foresT_train.csv")
test=pandas.read_csv("foresT_test.csv")

train.drop('Id',axis=1,inplace=True)
ids=test['Id']
test.drop('Id',axis=1,inplace=True)
labels_train=train['Cover_Type']
train.drop('Cover_Type',axis=1,inplace=True)
model = Sequential()
model.add(Dense(512, input_dim=54))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Activation('softmax'))
labels_train = np_utils.to_categorical(labels_train)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
clf=model
#clf = linear_model.LogisticRegression(C=1e5)
# clf = svm.SVC(kernel='linear', C=10000)
#clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf.fit(np.array(train),labels_train,validation_split=0.1,epochs=88)
out = clf.predict_classes(np.array(test))

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

