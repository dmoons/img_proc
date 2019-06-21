# -*- coding: utf-8 -*-
import os
import numpy as np

def img2vector(filename, label): #图像数据转为向量
    f = open(filename,'r')
    row_data = f.read()
    row_data = row_data.replace('\n','')  #换行符转为空格
    row_data = row_data + label
    row_data = np.array(map(int, list(row_data)))  #将string转为np.array
    return row_data

#k紧邻(KNN)分类算法
def classify0(rowX, dataSet, k):
    '''
    rowX是待分类的向量, dataSet是标记好的训练集, k表示选择最近邻居的数目
    '''
    #距离计算：绝对值距离
    dataSetSize = dataSet.shape[0]
    #print dataSetSize
    rowMat = np.zeros((dataSetSize, 1025), np.int)
    for i in range(dataSetSize):
        rowMat[i] = rowX
    diffMat = rowMat - dataSet
    label0 = dataSet[:,1024]         #取出训练集label
    diffMat2 = diffMat[:,0:1024]    #差分矩阵去除label列
    diffMat3 = diffMat2**2  #差分矩阵的平方，即是绝对值
    dis = diffMat3.sum(axis = 1)  #沿行求和，即是该待分类向量与训练集中每条数据的距离

    #选择距离最小的k个点
    sortedIndice = dis.argsort()
    #print sortedIndice
    vote_label = np.zeros((1,10), np.int)

    for i in range(k):
        label= label0[sortedIndice[i]] #获取第i小距离的label
        vote_label[0,label] = vote_label[0,label] + 1
    sorted_vote = vote_label.argsort()
    #print sorted_vote
    return sorted_vote[0,9]

#将训练集数据存储到np数组train_data中
train_dir = 'trainDigits'
train_filename = os.listdir(train_dir)  #获取trainingDigits目录下的文件名
m = len(train_filename)
train_data = np.zeros((m,1025), np.int)
for i in range(0, m):
    label = train_filename[i].split('_')[0]
    row = img2vector(train_dir +  train_filename[i], label)
    train_data[i] = row

#将测试集数据存储到np数组test_data中
test_dir = 'testDigits'
test_filename = os.listdir(test_dir)  #获取trainingDigits目录下的文件名
m = len(test_filename)
test_data = np.zeros((m,1025), np.int)
test_result = np.zeros((m,1),np.int)
for i in range(0, m):
    label = test_filename[i].split('_')[0]
    test_result[i] = int(label)  #存储测试集正确的分类
    row = img2vector(test_dir +  test_filename[i], '0')   #测试集初始分类设置为0
    test_data[i] = row

cc = 0

for i in range(m):
    ll = classify0(test_data[i], train_data, 5)
    #print ll,test_result[i]
    if ll == test_result[i]:
        cc = cc + 1
    else:
        print(i, ll, test_result[i])

print ('正确率是：%f' %(float(cc)/float(m)))

# -------------------------------------------------------------
# 原文：https://blog.csdn.net/zhulf0804/article/details/53843323

