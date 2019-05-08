#!/usr/local/bin/python3
# -*-coding:utf-8 -*-

import cv2
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv2.imread("./images/6789.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


gauss = cv2.GaussianBlur(gray, (3, 3), 1)

maxvalue = 255

def onaptivethreshold(x):
    value = cv2.getTrackbarPos("value", "Threshold")
    if(value < 3):
        value = 3
    if(value % 2 == 0):
        value = value + 1
    args = cv2.adaptiveThreshold(gauss, maxvalue, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, value, 1)
    gaus = cv2.adaptiveThreshold(gauss, maxvalue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, value, 1)
    cv2.imshow("Args", args)
    cv2.imshow("Gaus", gaus)

    refCnts, hierarchy = cv2.findContours(gauss.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]#排列轮廓，没意义



cv2.namedWindow("Threshold")

cv2.createTrackbar("value", "Threshold", 0, 10, onaptivethreshold)

cv2.imshow("Threshold", img)

cv2.waitKey(0)


'''
edges = cv.Canny(img, 100, 200)
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
'''

'''
imgSplit = cv2.split(img)
flag, b = cv2.threshold(imgSplit[2], 0, 255, cv2.THRESH_OTSU) 

element = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,1))
cv2.erode(b,element)

edges = cv2.Canny(b, 150, 200, 3, 5)


im = img.copy()

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 275, minLineLength = 600, maxLineGap = 100)[0]

for x1,y1,x2,y2 in lines:
    for index, (x3,y3,x4,y4) in enumerate(lines):
        if y1==y2 and y3==y4: # Horizontal Lines
            diff = abs(y1-y3)
        elif x1==x2 and x3==x4: # Vertical Lines
            diff = abs(x1-x3)
        else:
            diff = 0

        if diff < 10 and diff is not 0:
            del lines[index]

gridsize = (len(lines) - 2) / 2

for x1,y1,x2,y2 in lines:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)

cv2.imshow('houghlines', im)
print("show hughlines")
cv2.waitKey(0)

cv2.destroyAllWindows()
'''
