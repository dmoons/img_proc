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

def show_image(img, title):
    plt.figure("Image") # 图像窗口名称
    plt.imshow(img, title)
    plt.axis('off') # 关掉坐标轴为 off
    plt.title(title) # 图像题目
    plt.show()

def to_contours_image(contours, ref_image):
    blank_background = np.zeros_like(ref_image)
    img_contours = cv2.drawContours(blank_background, contours, -1, (255, 255, 255), thickness=2)
    return img_contours

def filte_contours(contours):
    conts = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 40:
            conts.append(contours[i])

    return conts

def fourier_calc(img):
    #1、读取文件，灰度化

    #2、图像延扩
    h, w = img.shape[:2]
    new_h = cv2.getOptimalDFTSize(h)
    new_w = cv2.getOptimalDFTSize(w)
    right = new_w - w
    bottom = new_h - h
    nimg = cv2.copyMakeBorder(img, 0, bottom, 0, right, borderType=cv2.BORDER_CONSTANT, value=0)
    cv2.imshow('new image', nimg)

    #3、执行傅里叶变换，并过得频域图像
    f = np.fft.fft2(nimg)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift))

    #二值化
    magnitude_uint = magnitude.astype(np.uint8)
    ret, thresh = cv2.threshold(magnitude_uint, 11, 255, cv2.THRESH_BINARY)
    # print(ret)
    cv2.imshow('thresh', thresh)
    print(thresh.dtype)
    #霍夫直线变换
    lines = cv2.HoughLinesP(thresh, 2, np.pi/180, 160, minLineLength=40, maxLineGap=100)
    if lines is None:
        return  
    print("line cnt:", len(lines))

    #创建一个新图像，标注直线
    lineimg = np.ones(nimg.shape, dtype=np.uint8)
    lineimg = lineimg * 255

    piThresh = np.pi/180
    pi2 = np.pi/2
    print("piThresh:", piThresh)

    theta = 0
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
            print("line theta:", theta)

    angle = math.atan(theta)
    print("atan:", angle)
    angle = angle * (180 / np.pi)
    print("atan * 180:", angle)
    angle = (angle - 90)/(w/h)
    print("atan - 90:", angle)

    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imshow('line image', lineimg)
    cv2.imshow('rotated', rotated)

#======================================================================
# load the input image, resize it, and convert it to grayscale
image = cv2.imread("images/meter4.jpeg")
# image = cv2.imread("images/6789.jpeg")

image = imutils.resize(image, width=600)
img = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)


rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
#cv2.imshow('tophat', tophat)

# compute the Scharr gradient of the tophat image, then scale
# the rest back into the range [0, 255]
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

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

thresh = cv2.GaussianBlur(thresh, (3,3), 0)
cv2.imshow("gaus", thresh)


refCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("find total contours: ", len(refCnts))
if (len(refCnts) < 1):
    exit(0)

leftCnts = []
# 循环浏览轮廓，提取符合设定的对象
for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if area > 100:
        #if (h >= 30 and h <= 60 and w / h <= 4 and w / h > 2.5):
        leftCnts.append(c)
        img = cv2.rectangle(img, (x-1,y-1), (x+w+1,y+h+1), (0,0,255), 1)


print("After filte, total contours: ", len(leftCnts))
if (len(leftCnts) < 1):
    exit(0)

newImg = to_contours_image(leftCnts, gray)
cv2.imshow("newImg", newImg)

#fourier_calc(newImg)


cv2.waitKey(0)

exit(0)
# cv2.imshow("group", group)
# cv2.imshow("roi", roi)

minRect = []
for c in leftCnts:
    # 获得最小的矩形轮廓 可能带旋转角度
    rect = cv2.minAreaRect(c)
    minRect.append(rect)


newImg = img.copy()
for rect in minRect:
    if 1:
        #print(rect)
        # 计算最小区域的坐标
        box = cv2.boxPoints(rect)
        # 坐标规范化为整数
        box = np.int0(box)
        # 画出轮廓
        cv2.drawContours(newImg, [box], 0, (0, 0, 255), 3)

cv2.imshow("img", newImg)

cv2.waitKey(0)

exit(0)

#----------------------------------------------------

gauss = cv2.GaussianBlur(group, (3, 3), 1)
gaus = cv2.adaptiveThreshold(gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY_INV, 11, 15)
cv2.imshow("Gaus", gaus)

# detect the contours of each individual digit in the group,
# then sort the digit contours from left to right
contours = cv2.findContours(gaus.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]

#digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
print("find digitCnts:", len(contours))

# filte the noise
digitCnts = []
for (i, c) in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if area > 40:
        if (h / w < 2 and w / h <= 1):
            digitCnts.append(c)

print("after filter, left: ", len(digitCnts))
img3 = to_contours_image(digitCnts, gaus)
cv2.imshow('digits', img3)

new_img = Image.fromarray(roi)
code = pytesseract.image_to_string(new_img, lang='eng', config='--psm 7')
print("code = ", code)

# img2 = image.copy()
# for (i, c) in enumerate(digitCnts):
#     # compute the bounding box for the digit, extract it, and resize
#     # it to a fixed size
#     (x, y, w, h) = cv2.boundingRect(c)
#     #if (h >= 30 and h <= 60 and w / h <= 4 and w / h > 2.5):
#     img2 = cv2.rectangle(img2, (x - 1, y - 1), (x + w + 1, y + h + 1),
#                           (0, 0, 255), 2)


# cv2.imshow("group2", img2)

cv2.waitKey(0)


exit(0)

#-------------------------------------------------------------------------
#提取小轮廓：
boxes = []
for i in range(len(hierarchy[0])):
    if hierarchy[0][i][3] == 0:
        boxes.append(hierarchy[0][i])

img = thresh
#提取数字：
nm = []
for j in range(len(boxes)):
    if boxes[j][2] != -1:
        x, y, w, h = cv2.boundingRect(cnts[boxes[j][2]])
        print("x,y,w,h=", x, y, w, h)
        #在原图中框选各个数字
        img = cv2.rectangle(img, (x-1,y-1), (x+w+1,y+h+1), (0,0,255), 2)

cv2.imshow('img', img)
cv2.waitKey(0)

locs = []
# loop over the contours
for (i, c) in enumerate(cnts):
    # compute the bounding box of the contour, then use the
    # bounding box coordinates to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    print("ar:", ar, "w", w, "h", h)

    # since credit cards used a fixed size fonts with 4 groups
    # of 4 digits, we can prune potential contours based on the
    # aspect ratio根据每个轮廓的宽高比进行过滤
    if ar > 2.5 and ar < 4.0:
        # contours can further be pruned on minimum/maximum width
        # and height使用纵横比，我们分析每个轮廓的形状。如果 ar 在2.5到4.0之间（比它高），
        # 以及40到55个像素之间的 w以及 10到20像素之间的h，我们将一个方便的元组的边界矩形参数附加到 locs
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            # append the bounding box region of the digits group
            # to our locations list
            locs.append((x, y, w, h))

# sort the digit locations from left-to-right, then initialize the
# list of classified digits
locs = sorted(locs, key=lambda x: x[0])
print("locs len:", len(locs))

# for cnt in range(len(locs)):
#     # 提取与绘制轮廓
#     cv2.drawContours(cardResult, cnts, cnt, (0, 255, 0), 1)

# cv2.imshow("Analysis Result", cardResult)
# #cv2.waitKey(0)
grpImgs = []
grpImgTitle = []
output = []
# loop over the 4 groupings of 4 digits
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # initialize the list of group digits
    groupOutput = []

    # extract the group ROI of 4 digits from the grayscale image,
    # then apply thresholding to segment the digits from the
    # background of the credit card
    group = gray[gY - 2:gY + gH + 2, gX - 2:gX + gW + 2]
    group = cv2.threshold(group, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    grpImgs.append(group)
    grpImgTitle.append("group" + str(i))

    # detect the contours of each individual digit in the group,
    # then sort the digit contours from left to right
    digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    #digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]
    #digitCnts = digitCnts[1]
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

    print("group ", i, " find ", len(digitCnts), " contours")

    # loop over the digit contours
    j = 0
    for c in digitCnts:
        # compute the bounding box of the individual digit, extract
        # the digit, and resize it to have the same fixed size as
        # the reference OCR-A images
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))

        grpImgs.append(roi)
        grpImgTitle.append("item" + str(j))

        print("--------- grp: ", i, " member: ", j, " -----------")
        #cv2.imshow("group digit " + str(j), roi)
        # cv2.waitKey(0)
        j = j + 1

        # initialize a list of template matching scores
        scores = []

        # loop over the reference digit name and digit ROI
        for (digit, digitROI) in digits.items():
            # apply correlation-based template matching, take the
            # score, and update the scores list
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (min_val, score, min_loc, max_loc) = cv2.minMaxLoc(result)
            scores.append(score)
            # top_left = max_loc
            # w, h = digitROI.shape[::-1]
            # bottom_right = (top_left[0] + w, top_left[1] + h)
            # cv2.rectangle(image, top_left, bottom_right, 255, 2)
            print("score:", score, ", digit:", digit)

        #cv2.imshow("group" + str(i) + ", digit " + str(j), roi)
        #cv2.imshow("matched digit " + str(np.argmax(scores)),
        #           digits[np.argmax(scores)])
        #cv2.waitKey(0)

        # the classification for the digit ROI will be the reference
        # digit name with the *largest* template matching score
        groupOutput.append(str(np.argmax(scores)))  # draw the digit classifications around the group
        cv2.rectangle(image, (gX - 5, gY - 5),
                      (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
        cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1)

    for k in range(len(digits)):
        grpImgs.append(digits[k])
        grpImgTitle.append("Temp" + str(k))

    for k in range(len(grpImgs)):
        plt.subplot(3, 5, k + 1), plt.imshow(grpImgs[k], 'gray')
        plt.title(grpImgTitle[k])
        plt.xticks([]), plt.yticks([])


    plt.show()

    grpImgTitle = []
    grpImgs = []

    # update the output digits list
    output.extend(groupOutput)

    cv2.imshow("group", group)
    #cv2.waitKey(0)

# display the output credit card information to the screen
#print("Credit Card Type: {}".format(FIRST_NUMBER.get(output[0], 'None')))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)  # TODO 效果不是很好，需要改进
cv2.waitKey(0)
