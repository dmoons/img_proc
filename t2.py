#!/usr/bin/env python3
# -*-coding:utf-8 -*-

import cv2
import os
import time

print(cv2.__version__)

# 保存截图
save_path = '/tmp/'

cap = cv2.VideoCapture(0)        #打开摄像头

# get a frame
ret, frame = cap.read()

# show a frame
# cv2.imshow("capture", frame)     #生成摄像头窗口

cv2.imwrite("./pic.png", frame)

# 转灰度图
gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imwrite("./gray.png", gray_img)

imageVar = cv2.Laplacian(gray_img, cv2.CV_64F).var()
print("", imageVar)

#获取灰度图矩阵的行数和列数
r,c = gray_img.shape[:2];
dark_sum = 0;	#偏暗的像素 初始化为0个
dark_prop = 0;	#偏暗像素所占比例初始化为0
piexs_sum = r*c;	#整个弧度图的像素个数为r*c

#遍历灰度图的所有像素
for row in gray_img:
    for colum in row:
        if colum < 40:	#人为设置的超参数,表示0~39的灰度值为暗
            dark_sum += 1;

dark_prop = dark_sum / (piexs_sum);
print("dark_sum:  " + str(dark_sum));
print("piexs_sum: " + str(piexs_sum));
print("dart_prop: " + str(dark_prop));
if dark_prop >= 0.75:	#人为设置的超参数:表示若偏暗像素所占比例超过0.78,则这张图被认为整体环境黑暗的图片
    print("pic is dark!");
else:
    print("pic is bright!")

cap.release()
cv2.destroyAllWindows()

