# -*- coding: utf-8 -*-

# import the necessary packages
from imutils import contours
import numpy as np
from matplotlib import pyplot as plt
import imutils
import cv2


# load the reference OCR-A image from disk, convert it to grayscale,
# and threshold it, such that the digits appear as *white* on a
# *black* background and invert it, such that the digits appear as *white* on a *black*
ref = cv2.imread("images/OCR-A.png")
h, w, ch = ref.shape
result = np.zeros((h, w, ch), dtype=np.uint8)

ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ret, ref = cv2.threshold(ref, 80, 255, cv2.THRESH_BINARY)
#ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

# find contours in the OCR-A image (i.e,. the outlines of the digits)
# sort them from left to right, and initialize a dictionary to map
# digit name to the ROI
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# for cnt in range(len(refCnts)):
#     # 提取与绘制轮廓
#     cv2.drawContours(ref, refCnts, cnt, (0, 255, 0), 1)

# cv2.imshow("Analysis Result", ref)
# cv2.waitKey(0)
# exit(0)

'''
#refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
print('len cnt:', len(refCnts))
refCont = cv2.drawContours(ref.copy(), refCnts, -1, (0, 255, 0), 1)
cv2.imshow('refCont', refCont)
# cv2.waitKey(0)
# exit(0)
'''

#refCnts = refCnts[1]
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]#排列轮廓，没意义
#print('sort_contours len cnt:',len(refCnts))

digits = {}


# 循环浏览轮廓，提取ROI并将其与相应的数字相关联
# loop over the OCR-A reference contours
for (i, c) in enumerate(refCnts):
    # compute the bounding box for the digit, extract it, and resize
    # it to a fixed size
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    # cv2.imshow('roi', roi)
    # cv2.waitKey(500)

    # update the digits dictionary, mapping the digit name to the ROI
    digits[i] = roi

# 从参考图像中提取数字，并将其与相应的数字名称相关联
print('digits:',digits.keys())

'''
#images = [ref]
# try 1
rows, cols = ref.shape
per = int(cols / 10)
for x in range(10):
    roi = ref[:, x * per:(x + 1) * per]
    roi = cv2.resize(roi, (57, 88))
    # cv2.imshow('roi', roi)
    # cv2.waitKey(0)

    # update the digits dictionary, mapping the digit name to the ROI
    digits[x] = roi
    #images.append(roi)
# 从参考图像中提取数字，并将其与相应的数字名称相关联
print('digits:', digits.keys())
'''

# image = cv2.imread("images/5.png")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ret, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# gray = imutils.resize(gray, 57, 88)

# rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
# sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

# compute the Scharr gradient of the tophat image, then scale
# the rest back into the range [0, 255]
# gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
# gradX = np.absolute(gradX)
# (minVal, maxVal) = (np.min(gradX), np.max(gradX))
# gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
# gradX = gradX.astype("uint8")

# apply a closing operation using the rectangular kernel to help
# cloes gaps in between credit card number digits, then apply
# Otsu's thresholding method to binarize the image
# gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
# thresh = cv2.threshold(gradX, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# images.append(image)
# images.append(gray)
# images.append(tophat)
# images.append(thresh)


# titles = [
#     'Refs', 'ref 0', 'ref 1', 'ref 2', 'ref 3', 'ref4', 'ref 5', 'ref 6',
#     'ref 7', 'ref 8', 'ref 9', 'input', 'gray', 'tophat', 'thresh'
# ]

# for i in range(len(images)):
#     plt.subplot(4, 4, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])

# plt.show()

# cv2.waitKey(0)
# exit(0)

###################################################################

# 初始化一对结构化的内核：
# 您可以将内核看作是一个小矩阵，我们在图像上滑动以进行（卷积）操作，例如模糊，锐化，边缘检测或其他图像处理操作。
# initialize a rectangular (wider than it is tall) and square
# structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 读取信用卡相片
# load the input image, resize it, and convert it to grayscale
image = cv2.imread("images/creditcard.png")
image = imutils.resize(image, width=300)

h, w, ch = image.shape
cardResult = np.zeros((h, w, ch), dtype=np.uint8)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply a tophat (whitehat) morphological operator to find light
# regions against a dark background (i.e., the credit card numbers)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
# cv2.imshow('tophat', tophat)
#cv2.waitKey(0)

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
# cv2.imshow('thresh 1', thresh)


# apply a second closing operation to the binary image, again
# to help close gaps between credit card number regions
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
# cv2.imshow('thresh 2', thresh)

images = [image]
images.append(gray)
images.append(tophat)
images.append(gradX1)
images.append(gradX)
images.append(thresh)

titles = ['image', 'gray', 'tophat', 'gradX1', 'gradX', 'thresh']

for i in range(len(images)):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

# # 给出一个矩形矩阵:
# k = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
# thresh = cv2.dilate(thresh, k)
# cv2.imshow('thresh 3', thresh)
# cv2.waitKey(0)


# find contours in the thresholded image, then initialize the
# list of digit locations找到轮廓并初始化数字分组位置列表。
cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
print("cnts = ", len(cnts))

# for cnt in range(len(cnts)):
#     # 提取与绘制轮廓
#     cv2.drawContours(cardResult, cnts, cnt, (0, 255, 0), 1)

#cv2.imshow("Contours Result", cardResult)
#cv2.waitKey(0)
#exit(0)


# h, w, ch = thresh.shape
# result = np.zeros((h, w, ch), dtype=np.uint8)
# for cnt in range(len(cnts)):
#     # 提取与绘制轮廓
#     cv2.drawContours(thresh, cnts, cnt, (0, 255, 0), 2)

'''
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
'''

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
