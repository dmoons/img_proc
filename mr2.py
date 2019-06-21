# -*- coding: utf-8 -*-

# import the necessary packages
from imutils import contours
import numpy as np
from matplotlib import pyplot as plt
import imutils
import cv2
import pytesseract
from PIL import Image
import math

def to_contours_image(contours, ref_image):
    blank_background = np.zeros_like(ref_image)
    img_contours = cv2.drawContours(blank_background, contours, -1, (255, 255, 255), thickness=2)
    return img_contours


# load the input image, resize it, and convert it to grayscale
# image = cv2.imread("images/6789.png")
image = cv2.imread("images/6789.jpeg")
image = imutils.resize(image, width=600)

img = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 6))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

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
thresh = cv2.threshold(gradX, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# apply a second closing operation to the binary image, again
# to help close gaps between digit number regions
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

refCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("find total contours: ", len(refCnts))
if (len(refCnts) < 1):
    exit(0)

leftCnts = []
roi = []
# 循环浏览轮廓，提取符合设定的对象
for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if area > 100:
        if (h >= 30 and h <= 60 and w / h <= 4 and w / h > 2.5):
            leftCnts.append(c)
            img = cv2.rectangle(img, (x-1,y-1), (x+w+1,y+h+1), (0,0,255), 1)
            roi = image[y:y + h, x - 2:x + w + 2]

print("After filte, total contours: ", len(leftCnts))
if (len(leftCnts) < 1):
    exit(0)


roi = imutils.resize(roi, height=58)

roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
roi_thresh = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#定义结构元素#闭运算
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
closed = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, kernel)
#显示腐蚀后的图像
roi_thresh = closed.copy()

gauss = cv2.GaussianBlur(roi_thresh, (3, 3), 1)
gaus = cv2.adaptiveThreshold(gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY_INV, 11, 15)

erode = cv2.erode(roi_thresh, kernel)

# detect the contours of each individual digit in the group,
# then sort the digit contours from left to right
contours = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]

#digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
print("find digitCnts:", len(contours))


new_img = Image.fromarray(erode)
code = pytesseract.image_to_string(new_img, config='--psm 7')
print("code = ", code)


