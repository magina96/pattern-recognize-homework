# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:48:59 2019

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
testset = pd.read_csv('E:\BaiduNetdiskDownload\mnist_test.csv')

datalabel = dataset.iloc[0:5000,0].values
datamat = dataset.iloc[0:5000,1:].values
testmat = testset.iloc[0:1000,:].values

def handwritting_classify(inx, dataset, labels, k):
    """inx是用于分类的输入向量，输入样本的训练集为dataset，标签向量是labels，k表示最近邻居的数目"""
    datamat_size = datamat.shape[0]
    diffmat = np.tile(inx,(datamat_size,1)) - datamat
    sqdiffmat = diffmaat ** 2
    #axis=1,表示将矩阵的每一行相加。
    sqdistances = sqdiffmat.sum(axis=1)
    distances = sqdistances ** 0.5
    #argsort()返回从小到大数值的索引值。
    sorted_distindicies = distances.argsort()
    classcount = {}

    for i in range(k):
        vote_label = labels[sorted_distindicies[i]]
        classCount[vote_label] = classCount.get(vote_label,0) + 1
    #从大到小排序
    sorted_classcount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_classcount[0][0]

def handwriting_classtest(trainfile, testfile):
    #读取训练，测试数据跟开头讲的方法一样，也可以写一个函数在这里调用
    #mtest是测试用的数据
    mtest = len(testmat)
    result = []

    for i in range(mtest):
        start_time = time.clock()
        result.append(handwritting_classify(testmat[i,:], datamat, datalabel, 3))
        circletime = time.clock() - start_time
        print("%d tasks lest,you need wait %.2f hours" % (mtest-1-i, (mtest-1-i)*circletime/3600))

    return result

result = handwriting_classtest('mnist_train.csv', 'mnist_test.csv')
submissions = pd.DataFrame({"Imageld":list(range(1,len(pred)+1)), "Label":pred})
submissions.to_csv("submissions.csv",index=False,header=True)
