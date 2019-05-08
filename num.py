#!/usr/local/bin/python3
# -*-coding:utf-8 -*-
import cv2
import os
import time

imgPath = "./images/shudu.jpg"
image = cv2.imread(imgPath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imwrite("./images/gray.jpg", gray)
#cv2.imshow("Image-gray",  gray)

# 对灰度图进行二值处理:
ret, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
#cv2.imshow("Image-thresh", thresh)

# 给出一个十字形矩阵:
k = cv2.getStructuringElement(cv2.MORPH_CROSS, (6, 6))

# 用k作为卷积内核，对二值图像进行膨胀处理：
d = cv2.dilate(thresh, k)

# 提取图片中的轮廓
im, co, h = cv2.findContours(d, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#提取小轮廓：
boxes = []
for i in range(len(h[0])):
    if h[0][i][3] == 0:
        boxes.append(h[0][i])

#提取数字：
nm = []
for j in range(len(boxes)):
    if boxes[j][2] != -1:
        x,y,w,h = cv2.boundingRect(co[boxes[j][2]])
        nm.append([x,y,w,h])
        #在原图中框选各个数字
        img = cv2.rectangle(img, (x-1,y-1), (x+w+1,y+h+1), (0,0,255), 2)

cv2.imwrite("./images/final.jpg", img)

