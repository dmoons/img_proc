#!/usr/local/bin/python3
# -*-coding:utf-8 -*-
import cv2
import os
import time

imgPath="./lightPicture/myImage.jpg"

image = cv2.imread(imgPath)

img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("./darkPicture/gray.jpg", img2gray)

img2HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imwrite("./darkPicture/hsv.jpg", img2HSV)

imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()

print("clarity:", imageVar)

#获取灰度图矩阵的行数和列数
r,c = img2gray.shape[:2];
dark_sum=0;	#偏暗的像素 初始化为0个
dark_prop=0;	#偏暗像素所占比例初始化为0
piexs_sum=r*c;	#整个弧度图的像素个数为r*c

#遍历灰度图的所有像素
for row in img2gray:
    for colum in row:
        if colum<40:	#人为设置的超参数,表示0~39的灰度值为暗
            dark_sum+=1;
dark_prop=dark_sum/(piexs_sum);
print("dark_sum:"+str(dark_sum));
print("piexs_sum:"+str(piexs_sum));
print("dark_prop=dark_sum/piexs_sum:"+str(dark_prop));
if dark_prop >=0.75:	#人为设置的超参数:表示若偏暗像素所占比例超过0.78,则这张图被认为整体环境黑暗的图片
    print("pic is dark!");
else:
    print("pic is bright!")
