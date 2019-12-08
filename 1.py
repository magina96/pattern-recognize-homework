# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:17:07 2019

@author: magina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn import svm
import operator
import time
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('E:\BaiduNetdiskDownload\mnist_train.csv')

images = dataset.iloc[0:5000,1:]
labels = dataset.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

i=3
test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])

clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)