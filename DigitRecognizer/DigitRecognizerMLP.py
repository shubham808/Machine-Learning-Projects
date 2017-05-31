from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.metrics import categorical_accuracy as accuracy
from keras import regularizers
import pandas as pd
import numpy as np

# Read data
train = pd.read_csv('train.csv')
labels_train = train.ix[:30000,0].values.astype('int32')
features_train = (train.ix[:30000,1:].values).astype('float32')
features_test = (train.ix[30001:,1:].values).astype('float32')
labels_test =train.ix[30001:,0].values.astype('int32')
test = (pd.read_csv('test.csv').values).astype('float32')
print(labels_train.shape)
print(labels_test.shape)

# convert list of labels to binary class matrix
labels_train = np_utils.to_categorical(labels_train)
labels_test = np_utils.to_categorical(labels_test)

# pre-processing: divide by max and substract mean
scale = np.max(features_train)
features_train /= 255
features_test /= 255
test/=scale
mean = np.std(features_train)
features_train -= mean
features_test -= mean
test-=mean
input_dim = features_train.shape[1]

nb_classes = labels_train.shape[1]
batch_size = 128
# Here's a Deep Dumb MLP (DDMLP)
model = Sequential()
model.add(Dense(512, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128,kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print("Training...")
model.fit(features_train, labels_train, validation_split=0.1, verbose=1)
print(model.evaluate(features_test,labels_test,batch_size=batch_size,verbose=0))
print("Generating test predictions...")
preds = model.predict_classes(test, verbose=0)
# 
# print('Accuracy: ', accuracy(labels_test, preds))
def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "keras-mlp.csv")
print("Done!")