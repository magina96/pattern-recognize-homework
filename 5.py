# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:24:26 2019

@author: magina
"""

from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pandas as pd
import numpy as np

pca = PCA(n_components=0.95, whiten=True)
digit = pd.read_csv('E:\BaiduNetdiskDownload\mnist_train_100.csv')
test = pd.read_csv('E:\BaiduNetdiskDownload\mnist_test_10.csv')
label = digit.values[:, 0].astype(int)
train = digit.values[:, 1:].astype(int)
test_data = test.values[:, :].astype(int)

pca.fit(train)
train_data = pca.transform(train)

svc = SVC()
svc.fit(train_data, label)

test_data = pca.transform(test_data)
ans = svc.predict(test_data)

a = []
for i in range(len(ans)):
    a.append(i+1)

np.savetxt('PCA_0.95_SVC.csv', np.c_[a, ans], 
    delimiter=',', header='ImageId,Label', comments='', fmt='%d')
