# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:43:11 2019

@author: magina
"""


import numpy as np
 
 
# 使用KNN分类
def kNNClassify(newInput, dataSet, labels, k):
    numSamples = len(dataSet)
 
    # 计算欧拉距离
    diff = np.tile(newInput, (numSamples, 1)) - dataSet
    squaredDiff = diff ** 2
    squaredDist = np.sum(squaredDiff, axis=1)
    distance = squaredDist ** 0.5
 
    # 距离排序
    sortedDistIndices = np.argsort(distance)
 
    # 计算前k个出现的次数
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
 
    # 找出最大的返回
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
 
    return maxIndex
 
 
def main():
    # 创建数据集
    def createDataSet():
        group = [[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]]
        labels = ['A', 'A', 'B', 'B']
        return group, labels
 
    dataSet, labels = createDataSet()
 
    testX = [1.2, 1.0]
    outputLabel = kNNClassify(testX, dataSet, labels, 3)
    print("测试数据:", testX, "被分类到: ", outputLabel)
 
    testX = [0.1, 0.3]
    outputLabel = kNNClassify(testX, dataSet, labels, 3)
    print("测试数据:", testX, "被分类到: ", outputLabel)
 
if __name__ == '__main__':
    exit(main())