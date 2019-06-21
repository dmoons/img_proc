# -*- coding: utf-8 -*-

# 将字符图片分离成字符
import os
import cv2
import numpy as np
import imutils
import math
import sys
from matplotlib import pyplot as plt

# 提取字符函数
def extract_chars(img):
    # 使用二值化处理
    ret, bwimg = cv2.threshold(img, 64, 255, cv2.THRESH_BINARY)
    cv2.imshow("threshold", bwimg)
    #cv2.waitKey(0)

    # 定义一个3*3的十字结构
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 对图像进行腐蚀,缩小边缘
    erode = cv2.erode(bwimg, kernel, iterations = 1)
    cv2.imshow("erode", erode)

    # 图像非运算，黑白颠倒
    #bw_image = cv2.bitwise_not(erode)

    # 检测图像轮廓，返回轮廓属性
    contours = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    # 生成轮廓框
    bounding_boxes = []
    for contour in contours:
        # 轮廓基点和长宽
        (x,y,w,h) = cv2.boundingRect(contour)
        (x,y,w,h) = x-2, y-2, w+4, h+4
        bounding_boxes.append((x,y,w,h))

    # 裁出字符图片
    characters = []
    for bbox in bounding_boxes:
        x,y,w,h = bbox
        char_image = img[y:y+h,x:x+w]
        characters.append(char_image)
    return characters


if __name__ =='__main__':
    # pics_path = sys.argv[1];#获取所给图片目录
    pics_path = "images/ARIAL.png"
    image = cv2.imread(pics_path)
    h,w,ch = image.shape
    print("imange shape:", image.shape)
    print("cnt = ", w/10)

    cv2.imshow("ARIAL.png", image)


    image2 = cv2.imread("images/0-9.jpg")
    h,w,ch = image2.shape
    print("imange shape:", image2.shape)
    print("cnt = ", h/10)

    cv2.imshow("0-9.jpg", image2)

    cv2.waitKey(0)
    exit(0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sub_img = extract_chars(gray)

    # fig = plt.figure()
    # for k in range(len(sub_img)):
    #     fig.add_subplot(2, 5, k + 1)
    #     plt.plot(sub_img[k + 1])

    # plt.show()

    # show image
    fig = plt.figure()
    plt.title("cut result")
    for i,img in enumerate(sub_img):
        fig.add_subplot(3,4,i+1)
        plt.imshow(img)
        #plt.imsave("./result/%s.jpg"%i,img)
    plt.show()


    cv2.imshow("image", image)
    cv2.waitKey(0)

