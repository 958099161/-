import time
import numpy as np

def loadData(fileName):
    dataArr = []; labelArr = []
    #读取文件
    fr = open(fileName)
    #遍历文件中的每一行
    for line in fr.readlines():
        curLine = line.strip().split(',')
        dataArr.append([int(int(num) > 128) for num in curLine[1:]])
        labelArr.append(int(curLine[0]))
    #返回数据集和标记
    return dataArr, labelArr

def majorClass(labelArr):
    classDict = {}
    #遍历所有标签
    for i in range(len(labelArr)):
        if labelArr[i] in classDict.keys():
            classDict[labelArr[i]] += 1
        else:
            #若无该标签，设初值为1，表示出现了1次了
            classDict[labelArr[i]] = 1
    #对字典依据值进行降序排序
    classSort = sorted(classDict.items(), key=lambda x: x[1], reverse=True)
    return classSort[0][0]

def calc_H_D(trainLabelArr):
    H_D = 0
    trainLabelSet = set([label for label in trainLabelArr])
    #遍历每一个出现过的标签
    for i in trainLabelSet:
        p = trainLabelArr[trainLabelArr == i].size / trainLabelArr.size
        #对经验熵的每一项累加求和
        H_D += -1 * p * np.log2(p)
    return H_D

def calcH_D_A(trainDataArr_DevFeature, trainLabelArr):
    H_D_A = 0
    trainDataSet = set([label for label in trainDataArr_DevFeature])
    for i in trainDataSet:
        H_D_A += trainDataArr_DevFeature[trainDataArr_DevFeature == i].size / trainDataArr_DevFeature.size \
                * calc_H_D(trainLabelArr[trainDataArr_DevFeature == i])
    return H_D_A

def calcBestFeature(trainDataList, trainLabelList):
    trainDataArr = np.array(trainDataList)
    trainLabelArr = np.array(trainLabelList).T
    featureNum = trainDataArr.shape[1]
    maxG_D_A = -1
    #初始化最大信息增益的特征
    maxFeature = -1
    #对每一个特征进行遍历计算
    for feature in range(featureNum):
        H_D = calc_H_D(trainLabelArr)
        trainDataArr_DevideByFeature = np.array(trainDataArr[:, feature].flat)
        G_D_A = H_D - calcH_D_A(trainDataArr_DevideByFeature, trainLabelArr)
        if G_D_A > maxG_D_A:
            maxG_D_A = G_D_A
            maxFeature = feature
    return maxFeature, maxG_D_A


def getSubDataArr(trainDataArr, trainLabelArr, A, a):
    retDataArr = []
    #返回的标签集
    retLabelArr = []
    #对当前数据的每一个样本进行遍历
    for i in range(len(trainDataArr)):
        #如果当前样本的特征为指定特征值a
        if trainDataArr[i][A] == a:
            #那么将该样本的第A个特征切割掉，放入返回的数据集中
            retDataArr.append(trainDataArr[i][0:A] + trainDataArr[i][A+1:])
            #将该样本的标签放入返回标签集中
            retLabelArr.append(trainLabelArr[i])
    #返回新的数据集和标签集
    return retDataArr, retLabelArr

def createTree(*dataSet):
    #该值的大小在设置上并未考虑太多，观察到信息增益前期在运行中为0.3左右，所以设置了0.1
    Epsilon = 0.1
    trainDataList = dataSet[0][0]
    trainLabelList = dataSet[0][1]
    #打印信息：开始一个子节点创建，打印当前特征向量数目及当前剩余样本数目
    print('start a node', len(trainDataList[0]), len(trainLabelList))
    classDict = {i for i in trainLabelList}
    if len(classDict) == 1:
        return trainLabelList[0]
    if len(trainDataList[0]) == 0:
        #返回当前标签集中占数目最大的标签
        return majorClass(trainLabelList)
    Ag, EpsilonGet = calcBestFeature(trainDataList, trainLabelList)
    if EpsilonGet < Epsilon:
        return majorClass(trainLabelList)
    treeDict = {Ag:{}}
    treeDict[Ag][0] = createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 0))
    treeDict[Ag][1] = createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 1))

    return treeDict

def predict(testDataList, tree):
    while True:
        (key, value), = tree.items()
        #如果当前的value是字典，说明还需要遍历下去
        if type(tree[key]).__name__ == 'dict':
            dataVal = testDataList[key]
            del testDataList[key]
            tree = value[dataVal]
            if type(tree).__name__ == 'int':
                return tree
        else:
            return value

def test(testDataList, testLabelList, tree):
    errorCnt = 0
    for i in range(len(testDataList)):
        if testLabelList[i] != predict(testDataList[i], tree):
            errorCnt += 1
    return 1 - errorCnt / len(testDataList)

if __name__ == '__main__':
    #开始时间
    start = time.time()
    # 获取训练集
    trainDataList, trainLabelList = loadData('../Mnist/mnist_train.csv')
    # 获取测试集
    testDataList, testLabelList = loadData('../Mnist/mnist_test.csv')
    #创建决策树
    print('start create tree')
    tree = createTree((trainDataList, trainLabelList))
    print('tree is:', tree)
    #测试准确率
    print('start test')
    accur = test(testDataList, testLabelList, tree)
    print('the accur is:', accur)
    #结束时间
    end = time.time()
    print('time span:', end - start)
