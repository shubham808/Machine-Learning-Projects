#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:50:46 2017

@author: shubham
"""
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn import cluster
from sklearn.svm import SVC
##---(Wed May 24 13:32:30 2017)---
from sklearn.datasets import load_digits
digits = load_digits()
import matplotlib.pyplot as plt
n_row, n_col = 2, 5
def print_digits(images,y,max=10):
    fig = plt.figure(figsize=(2.*n_col,2.26*n_col))
    i=0
    while i < max and i < images.shape[0]:
        p=fig.add_subplot(n_row,n_col,i+1)
        p.imshow(images[i],cmap=plt.cm.bone,interpolation='nearest')
        p.text(0, -1, str(y[i]))
        i = i + 1
        
#print_digits(digits.images, digits.target, max=10)
    

pca=PCA(n_components=2)
X_pca=pca.fit_transform(digits.data)

def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white',
                'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        px = X_pca[:,0][digits.target == i]
        py = X_pca[:,1][digits.target == i]
        plt.scatter(px, py, c=colors[i])
    plt.legend(digits.target_names)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    
plot_pca_scatter()


x_train,x_test,y_train,y_test,images_train,images_test=train_test_split(digits.data,digits.target,digits.images,test_size=0.25,random_state=42)



clf = cluster.KMeans(init='k-means++',n_clusters=10, random_state=42)
clf = SVC()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(y_test)
right=[]
for i in range(len(y_pred)):
    if y_test[i]==y_pred[i]:
        right.append(y_test[i])
print(np.array(right))        
sys.exit()    
def print_cluster(images, y_pred, cluster_number):
    images = images[y_pred==cluster_number]
    y_pred = y_pred[y_pred==cluster_number]
    print_digits(images, y_pred,max=10)
    
for i in range(10):
    print_cluster(images_test,y_pred,i)

pca = PCA(n_components=2).fit(x_train)
reduced_X_train = pca.transform(x_train)
h = .01

x_min, x_max = reduced_X_train[:, 0].min() + 1,reduced_X_train[:, 0].max() - 1
y_min, y_max = reduced_X_train[:, 1].min() + 1,reduced_X_train[:, 1].max() - 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
kmeans = cluster.KMeans(init='k-means++', n_clusters=10,n_init=10)
kmeans.fit(reduced_X_train)
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z=Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(),yy.max()),cmap=plt.cm.Paired, aspect='auto', origin='lower')
plt.plot(reduced_X_train[:, 0], reduced_X_train[:, 1], 'k.',markersize=2)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],marker='.',s=169, linewidths=3, color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCAreduced data)\nCentroids are marked with white dots')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))
print(metrics.completeness_score(y_test, y_pred))
print(y_test)
for i in range(len(y_pred)):
    if y_test[i]==y_pred[i]:
        print(y_test[i])