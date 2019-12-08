import numpy as np
import pandas as pd
 
import kNN
 
 
# 加载数据
def loadDataSet():
    # 获取训练集
    print('获取训练集...')
 
    trainingFile = pd.read_csv('E:\BaiduNetdiskDownload\mnist_train.csv')
    train_x = np.array(trainingFile.drop('label', axis = 1))[:41900]
    train_x[train_x > 0] = 1
    train_y = np.array(trainingFile['label'])[:41900]
 
    # 获取测试集
    print('获取测试集...')
 
    testingFile = pd.read_csv('E:\BaiduNetdiskDownload\mnist_train.csv')
    test_x = np.array(testingFile.drop('label', axis = 1))[41900:]
    test_x[test_x > 0] = 1
    test_y = np.array(testingFile['label'])[41900:]
 
    return train_x, train_y, test_x, test_y
 
# 手写数字测试
def testHandWritingClass():
    # 加载数据
    print('加载数据...')
 
    train_x, train_y, test_x, test_y = loadDataSet()
 
    # 训练
    print('训练中...')
 
    pass
 
    # 测试
    print('测试中...')
 
    numTestSamples = len(test_x)
    matchCount = 0
    result = []
    for i in range(numTestSamples):
        predict = kNN.kNNClassify(test_x[i], train_x, train_y, 3)
        if predict == test_y[i]:
            matchCount += 1
 
    accuracy = float(matchCount) / numTestSamples
 
    # 输出结果
    print('输出结果...')
 
    print('分类准确率为: %.2f%%' % (accuracy * 100))
 
if __name__ == '__main__':
    testHandWritingClass()
