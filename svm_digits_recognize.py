# -*- coding: utf-8 -*-
import os
from sklearn import svm
 
def img2vector(filename): #图像数据转为list
    f = open(filename,'r')
    row_data = f.read()
    row_data = row_data.replace('\n','')  #换行符转为空格
    row_data = list(row_data)
    for i in range(len(row_data)):
        row_data[i] = int(row_data[i])
    return row_data
 
train_dir = 'trainingDigits\\'   
train_filename = os.listdir(train_dir)  #获取trainingDigits目录下的文件名
m = len(train_filename)
 
X = []
Y = []
for i in range(0, m):
    label = train_filename[i].split('_')[0]
    Y.append(int(label))
    row = img2vector(train_dir +  train_filename[i])
    X.append( row )
 
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y) 
 
test_dir = 'testDigits\\'   
test_filename = os.listdir(test_dir)  #获取trainingDigits目录下的文件名
m = len(test_filename)
X_test = []
Y_test = []
 
for i in range(0, m):
    label = test_filename[i].split('_')[0]
    Y_test.append(int(label))  #存储测试集正确的分类
    row = img2vector(test_dir +  test_filename[i])   
    X_test.append(row)
ans = clf.predict(X_test)
 
cc = 0
ll = len(ans)
for i in range(ll):
    if Y_test[i] == ans[i]:
        cc +=1
    else:
        print "分错的文件为%s,被分类为%d" %(test_filename[i],ans[i])
 
print '正确率是：%f' % (1.0*cc/ll)
# --------------------- 
# 作者：zhulf0804 
# 来源：CSDN 
# 原文：https://blog.csdn.net/zhulf0804/article/details/53843323 
# 版权声明：本文为博主原创文章，转载请附上博文链接！