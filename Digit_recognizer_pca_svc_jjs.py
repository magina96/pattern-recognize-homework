# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:31:34 2019

@author: magina
"""

from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pandas as pd
import numpy as np

pca = PCA(n_components=0.95, whiten=True)
a=np.loadtxt("E:\BaiduNetdiskDownload\mnist_train.csv",dtype=int,delimiter=',')#加载数据为ndarray
digit=a[np.arange(0,42000)]        #提取多行
digit = pd.DataFrame(digit)
b=np.loadtxt("E:\BaiduNetdiskDownload\mnist_test.csv",dtype=int,delimiter=',')#加载数据为ndarray
test1=b[np.arange(0,10000)]        #提取多行
test = pd.DataFrame(test1)

label = digit.values[:, 0].astype(int)
train = digit.values[:, 1:].astype(int)
test_data = test.values[:, 1:].astype(int)

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
test1=test1[:,0]
n = 0
acry = 0
for n in range(10000):

 if test1[n,] == ans[n,]:
    acry =acry+1 
else: 
    acry =acry+0
print(acry/10000)