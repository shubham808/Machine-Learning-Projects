
import sys
from keras.utils import np_utils
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from gensim import corpora
from keras.layers import LSTM
from keras.datasets import imdb
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
max_features = 20000
maxlen = 80  
batch_size = 32

train=pd.read_csv('train_imdb.tsv',sep='\t',header=0)
test=pd.read_csv('test_imdb.tsv',sep='\t',header=0)

y_train=train['Sentiment'].values

x_train_raw=train['Phrase'].values
x_test_raw=test['Phrase'].values

stop_words=set(stopwords.words('english'))
stop_words.update(['!','#','.',',','"',"'",'(',')','[',']','{','}',':',';']) 

stemmer=SnowballStemmer('english')

com_train=[]
for x in x_train_raw:
    x=word_tokenize(x)
    f=[i for i in x if i not in stop_words and i.isalpha()]
    x=[stemmer.stem(i) for i in f]
    com_train.append(x)
com_test=[]
for x in x_test_raw:
    x=word_tokenize(x)
    f=[i for i in x if i not in stop_words and i.isalpha()]
    x=[stemmer.stem(i) for i in f]
    com_test.append(x)    

combined=np.concatenate((com_train,com_test),axis=0)

dictionary=corpora.Dictionary(combined)              

word_train_id=[]
for word in com_train:
    word_ids=[dictionary.token2id[w] for w in word]
    word_train_id.append(word_ids)


word_test_id=[]
for word in com_test:
    word_ids=[dictionary.token2id[w] for w in word]
    word_test_id.append(word_ids)

x_train=sequence.pad_sequences(np.array(word_train_id),maxlen=500)
x_test=sequence.pad_sequences(np.array(word_test_id),maxlen=500)
y_train=np_utils.to_categorical(y_train)

with tf.device('/gpu:2'):
    model = Sequential()
    model.add(Embedding(len(dictionary.keys()), 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    
    model.fit(x_train, y_train,batch_size=batch_size,epochs=15,validation_split=0.1)
    
    prediction=model.predict_classes(x_test)
    
    #write
    writer = csv.writer(sys.stdout)
    writer.writerow(("PhraseId", "Sentiment"))
    for datapoint, sentiment in zip(test, prediction):
        writer.writerow((datapoint.Phraseid, sentiment))
