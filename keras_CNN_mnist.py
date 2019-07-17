# -*- coding: utf-8 -*-

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam#导入SGD优化器，Adam优化器
 
#载入数据
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#(60000,28,28)
#print('x_shape:',x_train.shape)
#(60000)
#print('y_shape:',y_train.shape)
 
#(60000,28,28)->(60000,28,28,1)
x_train=x_train.reshape(-1,28,28,1)/255.0 #除以255是归一化
x_test=x_test.reshape(-1,28,28,1)/255.0
#换one_hot 格式，把像素点转变成0、1形式
y_train=np_utils.to_categorical(y_train,num_classes=10)#把y_train分成10个类别
y_test=np_utils.to_categorical(y_test,num_classes=10)
 
#定义模型
model=Sequential()
 
#定义第一个卷积层
#input_shape输入平面
#filters 卷积核/滤波器个数
#kernel_size 卷积窗口的大小
#strides步长
#padding padding方式same/valid
#activation激活函数
model.add(Convolution2D(
    input_shape=(28,28,1),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    activation='relu'
))
 
#第一个池化层
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same'
))
 
#第二个卷积层
model.add(Convolution2D(64,5,strides=1,padding='same',activation='relu'))
 
#第二个池化层
model.add(MaxPooling2D(2,2,'same'))
 
#把第二个池化层的输出扁平化为1维
model.add(Flatten())
 
#第一个全连接层
model.add(Dense(1024,activation='relu'))
 
#Dropout
model.add(Dropout(0.5))
 
#第二个全连接层
model.add(Dense(10,activation='softmax'))
 
#定义优化器
#sgd=SGD(lr=0.2)
adam=Adam(lr=0.001)#lr是学习率
 
#定义优化器，loss function，训练过程中计算准确率,二次代价函数改为categorical_crossentropy交叉熵函数
model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])#这里还可以计算准确率
 
#训练模型，可以用fit函数
model.fit(x_train,y_train,batch_size=64,epochs=10)#从60000张图中每次拿64张来训练，60000张图训练完叫一个周期，一共训练10个周期
 
#评估模型，用evaluate（）函数
loss,accuracy=model.evaluate(x_test,y_test)
print('\ntest loss',loss)
print('test accuracy',accuracy)
 
loss,accuracy=model.evaluate(x_train,y_train)
print('\ntrain loss',loss)
print('train accuracy',accuracy)
 
 

# --------------------- 
# 作者：iamcfb_ 
# 来源：CSDN 
# 原文：https://blog.csdn.net/iamcfb_/article/details/87548010 
# 版权声明：本文为博主原创文章，转载请附上博文链接！