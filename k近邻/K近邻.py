'''
数据集：Mnist
训练集数量：60000
测试集数量：实际使用：100
------------------------------
运行结果：（邻近k数量：25）
向量距离使用算法——欧式距离
    正确率：0.98
    运行时长：108.94
'''

import numpy as np
import time
from tqdm import tqdm

def loadData(fileName):
    '''
    加载Mnist数据集
    :param fileName:要加载的数据集路径
    :return: list形式的数据集及标记
    '''
    print('start to read data')
    data = []
    label = []
    fr = open(fileName, 'r')
    for line in fr.readlines():
        curLine = line.strip().split(',')
        # if int(curLine[0]) >= 5:
        #     label.append(1)
        # else:
        #     label.append(-1)
        label.append(int(curLine[0]))
        data.append([int(num)/255 for num in curLine[1:]])
    return data, label

# 向量之间的欧式距离
def calcDist(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2)))

def getClosest(trainDataMat, trainLabelMat, x, topK):
    '''
    :param trainDataMat:训练集数据集
    :param trainLabelMat:训练集标签集
    :param x:要预测的样本x
    :param topK:选择参考最邻近样本的数目（样本数目的选择关系到正确率，详看3.2.3 K值的选择）
    :return:预测的标记
    '''
    distList = [0] * len(trainLabelMat)
    for i in range(len(trainDataMat)):
        x1 = trainDataMat[i]
        curDist = calcDist(x1, x)
        distList[i] = curDist
    topKList = np.argsort(np.array(distList))[:topK]        #升序排序
    labelList = [0] * 10
    for index in topKList:
        labelList[int(trainLabelMat[index])] += 1
    return labelList.index(max(labelList))


def test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, topK):
    '''
    测试正确率
    :param trainDataArr:训练集数据集
    :param trainLabelArr: 训练集标记
    :param testDataArr: 测试集数据集
    :param testLabelArr: 测试集标记
    :param topK: 选择多少个邻近点参考
    :return: 正确率
    '''
    print('start test')
    #将所有列表转换为矩阵形式，方便运算
    trainDataMat = np.mat(trainDataArr);
    trainLabelMat = np.mat(trainLabelArr).T
    testDataMat = np.mat(testDataArr);  # 10000,784
    testLabelMat = np.mat(testLabelArr).T  # 10000,1
    #错误值技术
    errorCnt = 0
    #测试100个样本点，时间比较长，没有实现kd树
    for i in tqdm(range(100)):
        x = testDataMat[i]
        y = getClosest(trainDataMat, trainLabelMat, x, topK)
        if y != testLabelMat[i]:
            errorCnt += 1
            # print("y: {},  testLabelMat[i]{}".format(y, testLabelMat[i]))
    return 1 - (errorCnt / 100)

if __name__ == "__main__":
    start = time.time()

    #获取训练集
    trainDataArr, trainLabelArr = loadData('../Mnist/mnist_train.csv')
    #获取测试集
    testDataArr, testLabelArr = loadData('../Mnist/mnist_test.csv')
    #计算测试集正确率
    accur = test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, 25)
    #打印正确率
    print('accur is:%d'%(accur * 100), '%')
    end = time.time()
    #显示花费时间
    print('time span:', end - start)


