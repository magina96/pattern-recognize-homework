# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:16:57 2019

@author: magina
"""
import pandas as pd
import numpy as np


a=np.loadtxt("E:\BaiduNetdiskDownload\mnist_train.csv",dtype=int,delimiter=',')#加载数据为ndarray
b1=a[np.arange(0,5000)]        #提取多行

b=np.loadtxt("E:\BaiduNetdiskDownload\mnist_test.csv",dtype=int,delimiter=',')#加载数据为ndarray
b2=b[np.arange(0,1000)]        #提取多行