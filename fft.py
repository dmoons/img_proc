import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

def fourier_demo():
    #1、读取文件，灰度化
    img = cv2.imread('./images/text_segment.jpg')
    cv2.imshow('original', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)

    '''
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 6))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    #cv2.imshow('tophat', tophat)

    # compute the Scharr gradient of the tophat image, then scale
    # the rest back into the range [0, 255]
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")

    gradX1 = gradX
    # apply a closing operation using the rectangular kernel to help
    # cloes gaps in between credit card number digits, then apply
    # Otsu's thresholding method to binarize the image
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow('thresh 1', thresh)

    # apply a second closing operation to the binary image, again
    # to help close gaps between credit card number regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    cv2.imshow('thresh 2', thresh)

    gray = thresh.copy()
    '''

    #2、图像延扩
    h, w = img.shape[:2]
    new_h = cv2.getOptimalDFTSize(h)
    new_w = cv2.getOptimalDFTSize(w)
    right = new_w - w
    bottom = new_h - h
    nimg = cv2.copyMakeBorder(gray, 0, bottom, 0, right, borderType=cv2.BORDER_CONSTANT, value=0)
    cv2.imshow('new image', nimg)

    #3、执行傅里叶变换，并过得频域图像
    f = np.fft.fft2(nimg)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift))


    #二值化
    magnitude_uint = magnitude.astype(np.uint8)
    ret, thresh = cv2.threshold(magnitude_uint, 11, 255, cv2.THRESH_BINARY)
    print(ret)
    cv2.imshow('thresh', thresh)
    print(thresh.dtype)
    #霍夫直线变换
    lines = cv2.HoughLinesP(thresh, 2, np.pi/180, 30, minLineLength=40, maxLineGap=100)
    print(len(lines))

    #创建一个新图像，标注直线
    lineimg = np.ones(nimg.shape,dtype=np.uint8)
    lineimg = lineimg * 255

    piThresh = np.pi/180
    pi2 = np.pi/2
    print(piThresh)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(lineimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if x2 - x1 == 0:
            continue
        else:
            theta = (y2 - y1) / (x2 - x1)
        if abs(theta) < piThresh or abs(theta - pi2) < piThresh:
            continue
        else:
            print(theta)

    angle = math.atan(theta)
    print("atan:", angle)
    angle = angle * (180 / np.pi)
    print("atan * 180:", angle)
    angle = (angle - 90)/(w/h)
    print("atan - 90:", angle)

    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imshow('line image', lineimg)
    cv2.imshow('rotated', rotated)

fourier_demo()
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------
# 作者：云net
# 来源：CSDN
# 原文：https://blog.csdn.net/qq_36387683/article/details/80530709
# 版权声明：本文为博主原创文章，转载请附上博文链接！
