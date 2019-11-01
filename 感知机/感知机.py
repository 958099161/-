'''
数据集：Mnist
训练集数量：60000
测试集数量：10000
iter：30
------------------------------
运行结果：
正确率：0.8172（二分类）
运行时长：53.49
'''

import numpy as np
import time
from tqdm import tqdm

# 读取data和label， label[0---4]:0  [5--9]:1
# 对data做归一化
def loadData(fileName):
    '''
    加载Mnist数据集
    :param fileName:要加载的数据集路径
    :return: list形式的数据集及标记
    '''
    print('start to read data')
    data = []; label = []
    fr = open(fileName, 'r')
    for line in fr.readlines():
        curLine = line.strip().split(',')
        if int(curLine[0]) >= 5:
            label.append(1)
        else:
            label.append(-1)
        data.append([int(num)/255 for num in curLine[1:]])
    return data, label

def perceptron(dataArr, labelArr, iter=50):
    '''
    :param dataArr:训练集的数据 (list)
    :param labelArr: 训练集的标签(list)
    :param iter: 迭代次数，默认50
    :return: 训练好的w和b
    '''
    print('start to trans')
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    #获取数据矩阵的大小，为m*n
    m, n = np.shape(dataMat)
    w = np.zeros((1, np.shape(dataMat)[1]))
    b= 0
    h = 0.0001
    for k in tqdm(range(iter)):
        for i in range(m):   # 逐一遍历，梯度下降
            xi = dataMat[i]
            yi = labelMat[i]
            if -1 * yi * (w * xi.T + b) >= 0:
                w = w + h *  yi * xi
                b = b + h * yi
    return w, b


def test(dataArr, labelArr, w, b):
    '''
    测试准确率
    :param dataArr:测试集
    :param labelArr: 测试集标签
    :param w: 训练获得的权重w
    :param b: 训练获得的偏置b
    :return: 正确率
    '''
    print('start to test')
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    m, n = np.shape(dataMat)
    errorCnt = 0
    for i in range(m):
        xi = dataMat[i]
        yi = labelMat[i]
        result = -1 * yi * (w * xi.T + b)
        if result >= 0: errorCnt += 1
    accruRate = 1 - (errorCnt / m)
    return accruRate

if __name__ == '__main__':
    start = time.time()
    #获取训练集及标签
    trainData, trainLabel = loadData('../Mnist/mnist_train.csv')
    #获取测试集及标签
    testData, testLabel = loadData('../Mnist/mnist_test.csv')
    #训练获得权重
    w, b = perceptron(trainData, trainLabel, iter = 30)
    #进行测试，获得正确率
    accruRate = test(testData, testLabel, w, b)
    #获取当前时间，作为结束时间
    end = time.time()
    #显示正确率
    print('accuracy rate is:', accruRate)
    #显示用时时长
    print('time span:', end - start)

