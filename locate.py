# -*- coding: utf-8 -*-

# import the necessary packages
from imutils import contours
import numpy as np
from matplotlib import pyplot as plt
import imutils
import cv2
import math
import os
import sys

def to_contours_image(contours, ref_image):
    blank_background = np.zeros_like(ref_image)
    img_contours = cv2.drawContours(blank_background, contours, -1, (255, 255, 255), thickness=2)
    return img_contours


def locate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 6))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    cv2.imshow("tophat", tophat)

    # compute the Scharr gradient of the tophat image, then scale
    # the rest back into the range [0, 255]
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")

    # apply a closing operation using the rectangular kernel to help
    # cloes gaps in between digits, then apply
    # Otsu's thresholding method to binarize the image
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    cv2.imshow("gradX", gradX)


    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("thresh1", thresh)

    # apply a second closing operation to the binary image, again
    # to help close gaps between digit number regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    cv2.imshow("thresh2", thresh)

    refCnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # refCnts, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("find total contours: ", len(refCnts))
    if (len(refCnts) < 1):
        return None, None

    leftCnts = []
    roi = []
    img = image.copy()
    # 循环浏览轮廓，提取符合设定的对象
    for (i, c) in enumerate(refCnts):
        (x, y, w, h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > 100:
            img = cv2.rectangle(img, (x-1,y-1), (x+w+1,y+h+1), (0,0,255), 1)
            # if (h >= 30 and h <= 60 and w / h <= 4 and w / h > 2.5):
            if (w / h <= 4 and w / h > 2.5):
                leftCnts.append(c)
                roi = image[y:y + h, x - 2:x + w + 2]

    cv2.imshow("contours 1", img)
    #cv2.waitKey(0)

    print("After filte, total contours: ", len(leftCnts))
    if (len(leftCnts) < 1):
        return None, None

    return roi, leftCnts


if __name__ =='__main__':
    pics_path = sys.argv[1] #获取所给图片目录
    image = cv2.imread(pics_path)
    image = imutils.resize(image, width=600)
    roi, contours = locate(image)

    if roi is None or contours is None:
        print("digit region locate failed!")
        exit(0)

    # show the found roi
    cv2.imshow("roi", roi)

    # 循环浏览轮廓，提取符合设定的对象
    for (i, c) in enumerate(contours):
        # compute the bounding box for the digit, extract it, and resize
        # it to a fixed size
        (x, y, w, h) = cv2.boundingRect(c)
        image = cv2.rectangle(image, (x-1,y-1), (x+w+1,y+h+1), (0,0,255), 2)

    cv2.imshow("contours 2", image)

    cv2.waitKey(0)

