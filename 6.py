

from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pandas as pd
import numpy as np

pca = PCA(n_components=0.95, whiten=True)
digit = pd.read_csv('E:\BaiduNetdiskDownload\mnist_train.csv')
test = pd.read_csv('E:\BaiduNetdiskDownload\mnist_test.csv')
digit = digit[0:5000,:]
test  = test[5000:6000,:]
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
