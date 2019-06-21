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
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])

print(sys.path)

def show_image(img, title):
    plt.figure("Image") # 图像窗口名称
    plt.imshow(img, 'gray')
    plt.axis('off') # 关掉坐标轴为 off
    plt.title(title) # 图像题目
    plt.show()

# template image should be dark background and white character
def load_template(templ_image_path):
    # load the reference OCR-A image from disk, convert it to grayscale,
    # and threshold it, such that the digits appear as *white* on a
    # *black* background and invert it, such that the digits appear as *white* on a *black*
    ref = cv2.imread(templ_image_path)
    h, w, ch = ref.shape
    result = np.zeros((h, w, ch), dtype=np.uint8)

    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ret, ref = cv2.threshold(ref, 80, 255, cv2.THRESH_BINARY)
    #ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

    # find contours in the OCR-A image (i.e,. the outlines of the digits)
    # sort them from left to right, and initialize a dictionary to map
    # digit name to the ROI
    refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #refCnts = refCnts[1]
    refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]#排列轮廓，没意义
    #print('sort_contours len cnt:',len(refCnts))

    digits = {}

    # 循环浏览轮廓，提取ROI并将其与相应的数字相关联
    for (i, c) in enumerate(refCnts):
        # compute the bounding box for the digit, extract it, and resize
        # it to a fixed size
        (x, y, w, h) = cv2.boundingRect(c)
        roi = ref[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))

        # update the digits dictionary, mapping the digit name to the ROI
        digits[i] = roi

    # 从参考图像中提取数字，并将其与相应的数字名称相关联
    print('digits:', digits.keys())
    return digits

def locate(imge):
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

    refCnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("find total contours: ", len(refCnts))
    if (len(refCnts) < 1):
        return None, None

    leftCnts = []
    roi = []
    # 循环浏览轮廓，提取符合设定的对象
    for (i, c) in enumerate(refCnts):
        (x, y, w, h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > 100:
            if (h >= 30 and h <= 60 and w / h <= 4 and w / h > 2.5):
                leftCnts.append(c)
                roi = image[y:y + h, x - 2:x + w + 2]

    print("After filte, total contours: ", len(leftCnts))
    if (len(leftCnts) < 1):
        return None, None

    return roi, leftCnts


def bolder_eliminate(image):
    # image=cv2.imread(read_file,1) #读取图片 image_name应该是变量
    # img = cv2.medianBlur(image,5) #中值滤波，去除黑色边际中可能含有的噪声干扰
    b = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)          #调整裁剪效果
    #binary_image = b[1]               #二值图--具有三通道
    #binary_image = cv2.cvtColor(binary_image,cv2.COLOR_BGR2GRAY)
    binary_image = image.copy()
    print(binary_image.shape)       #改为单通道
 
    x=binary_image.shape[0]
    print("高度x=",x)
    y=binary_image.shape[1]
    print("宽度y=",y)
    edges_x=[]
    edges_y=[]
    for i in range(x):
        for j in range(y):
            if binary_image[i][j]==255:
                edges_x.append(i)
                edges_y.append(j)
 
    left = min(edges_x)               #左边界
    right = max(edges_x)              #右边界
    width = right-left + 1                #宽度
    bottom = min(edges_y)             #底部
    top = max(edges_y)                #顶部
    height = top - bottom               #高度
 
    pre1_picture=image[left:left+width, bottom:bottom+height]        #图片截取
    return pre1_picture                                             #返回图片数据

def character_splite(image):
    fig = plt.figure()
    fig.add_subplot(2,3,1)
    plt.title("raw image")
    plt.imshow(image)

    binary_image = image.copy()

    # counting non-zero value by row , axis y
    row_nz = []
    for row in binary_image.tolist():
        row_nz.append(len(row) - row.count(0))
    fig.add_subplot(2,3,3)
    plt.title("non-zero values on y (by row)")
    plt.plot(row_nz)

    # counting non-zero value by column, x axis
    col_nz = []
    for col in binary_image.T.tolist():
        col_nz.append(len(col) - col.count(0))
    fig.add_subplot(2,3,4)
    plt.title("non-zero values on y (by col)")
    plt.plot(col_nz)

    ##### start split
    # first find upper and lower boundary of y (row)
    fig.add_subplot(2,3,5)
    plt.title("y boudary deleted")
    upper_y = 0
    for i,x in enumerate(row_nz):
        if x != 0:
            upper_y = i
            break
    lower_y = 0
    for i,x in enumerate(row_nz[::-1]):
        if x!=0:
            lower_y = len(row_nz) - i
            break
    sliced_y_img = binary_image[upper_y:lower_y,:]
    plt.imshow(sliced_y_img)

    # then we find left and right boundary of every digital (x, on column)
    fig.add_subplot(2,3,6)
    plt.title("x boudary deleted")
    column_boundary_list = []
    record = False

    # the start boundary
    if col_nz[0] != 0:
        column_boundary_list.append(1)

    for i,x in enumerate(col_nz[:-1]):
        if (col_nz[i] == 0 and col_nz[i+1] != 0) or (col_nz[i] != 0 and col_nz[i+1] == 0):
            column_boundary_list.append(i+1)
            print("col ", i)

    # the end boundary
    if col_nz[-1] != 0:
        column_boundary_list.append(len(col_nz) + 1)

    img_list = []
    xl = [ column_boundary_list[i:i+2] for i in range(0,len(column_boundary_list),2) ]
    for x in xl:
        img_list.append(sliced_y_img[:,x[0]:x[1]] )

    print("cut img cnt:", len(img_list))

    # del invalid image
    digits = []
    for x in img_list:
        if x.shape[1] > 5:
            print("x shape:", x.shape)
            x = bolder_eliminate(x)
            #x = cv2.resize(x, (57, 88))
            digits.append(x)

    print("final img list:", len(digits))

    # show image
    fig = plt.figure()
    plt.title("cut result")
    for i,img in enumerate(digits):
        fig.add_subplot(3,4,i+1)
        plt.imshow(img)
        plt.imsave("./result/%s.jpg"%i,img)
    plt.show()

    return digits


def template_match(template, roi):
    scores = []
    # loop over the reference digit name and digit ROI
    for (digit, digitROI) in template.items():
        # apply correlation-based template matching, take the
        # score, and update the scores list
        result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
        (min_val, score, min_loc, max_loc) = cv2.minMaxLoc(result)
        scores.append(score)
        print("score:", score, ", digit:", digit)

    return scores

#================================================================
RefDigits = load_template("images/ARIAL.jpg")

# load the input image, resize it, and convert it to grayscale
image = cv2.imread("/Users/leon/Project/MeterRemoteReader/watermeter_pics/WechatIMG186.jpeg")
image = imutils.resize(image, width=600)
img = image.copy()

roi, contours = locate(image)
if roi is None or contours is None:
    print("failed to loate digit regin")
    exit(-1)

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# img = cv2.fastNlMeansDenoisingMulti(gray, 2, 5, None, 4, 7, 35)
# cv2.imshow("nosity", img)
# cv2.waitKey(0)

#gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
gauss = cv2.GaussianBlur(gray, (3, 3), 1)
#gaus = cv2.adaptiveThreshold(gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 1)
gaus = cv2.adaptiveThreshold(gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 15)
cv2.imshow("Gaus", gaus)
cv2.waitKey(0)

kernel = np.ones((2, 2), np.uint8)

# 腐蚀
erroding = cv2.erode(gaus, kernel)
#sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 6))
#erroding = cv2.morphologyEx(gaus.copy(), cv2.MORPH_CLOSE, sqKernel)
#erroding = cv2.erode(erroding, kernel)

cv2.imshow("errode", erroding)


# 膨胀
dilation = cv2.morphologyEx(gaus.copy(), cv2.MORPH_OPEN, kernel)
cv2.imshow("dilate", dilation)

# 去除孤立小区域
contours,hierarch = cv2.findContours(gaus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area < 20:
        # 将小区域涂黑
        cv2.drawContours(gaus, [contours[i]], 0, 0, -1)

cv2.imshow("clear", gaus)

roi_digits = character_splite(gaus)

# new_img = Image.fromarray(gaus)
# code = pytesseract.image_to_string(new_img, config='--psm 7')
# print("code = ", code)
output = []
groupOutput = []
for digit in roi_digits:
    scores = template_match(RefDigits, digit)
    groupOutput.append(str(np.argmax(scores)))  # draw the digit classifications around the group


# update the output digits list
output.extend(groupOutput)

# display the output credit card information to the screen
print("Meter digits: {}".format("".join(output)))
cv2.imshow("Image", image)  # TODO 效果不是很好，需要改进
cv2.waitKey(0)
