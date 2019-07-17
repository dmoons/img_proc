# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pytesseract
from PIL import Image

def initKnn():
    knn = cv2.ml.KNearest_create()
    img = cv2.imread('digits.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
    train = np.array(cells).reshape(-1, 400).astype(np.float32)
    trainLabel = np.repeat(np.arange(10), 500)
    return knn, train, trainLabel


def updateKnn(knn, train, trainLabel, newData=None, newDataLabel=None):
    if (newData is not None) and (newDataLabel is not None):
        print(train.shape, newData.shape)
        newData = newData.reshape(-1, 400).astype(np.float32)
        train = np.vstack((train, newData))
        trainLabel = np.hstack((trainLabel, newDataLabel))
    knn.train(train, cv2.ml.ROW_SAMPLE, trainLabel)
    return knn, train, trainLabel


def findRoi(frame, thresValue):
    rois = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.dilate(gray, None, iterations=2)
    gray2 = cv2.erode(gray2, None, iterations=2)
    edges = cv2.absdiff(gray, gray2)
    x = cv2.Sobel(edges, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(edges, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    ret, ddst = cv2.threshold(dst, thresValue, 255, cv2.THRESH_BINARY)

    cv2.imshow("thresh", ddst)
    cv2.waitKey()

    # contours, hierarchy = cv2.findContours(ddst, cv2.RETR_EXTERNAL,
    #                                            cv2.CHAIN_APPROX_SIMPLE)
    # for c in contours:
    #     x, y, w, h = cv2.boundingRect(c)
    #     if w > 10 and h > 20:
    #         rois.append((x, y, w, h))

    rois = extract_chars(frame)

    return rois, edges


def findDigit(knn, roi, thresValue):
    ret, th = cv2.threshold(roi, thresValue, 255, cv2.THRESH_BINARY)
    th = cv2.resize(th, (20, 20))
    out = th.reshape(-1, 400).astype(np.float32)
    ret, result, neighbours, dist = knn.findNearest(out, k=5)
    return int(result[0][0]), th


def concatenate(images):
    n = len(images)
    output = np.zeros(20 * 20 * n).reshape(-1, 20)
    for i in range(n):
        output[20 * i:20 * (i + 1), :] = images[i]

    return output


# 提取字符函数
def extract_chars(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用二值化处理
    ret, bwimg = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    #cv2.imshow("threshold", bwimg)
    # cv2.waitKey(0)

    # 定义一个3*3的十字结构
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 对图像进行腐蚀,缩小边缘
    erode = cv2.erode(bwimg, kernel, iterations=1)
    #cv2.imshow("erode", erode)

    # 图像非运算，黑白颠倒
    bw_image = cv2.bitwise_not(erode)
    #cv2.imshow("inverse", bw_image)

    # 检测图像轮廓，返回轮廓属性
    contours = cv2.findContours(bw_image, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)[0]
    print("contours find:", len(contours))

    # 生成轮廓框
    bounding_boxes = []
    for contour in contours:
        # 轮廓基点和长宽
        (x, y, w, h) = cv2.boundingRect(contour)
        (x, y, w, h) = x - 2, y - 2, w + 2, h + 2
        bounding_boxes.append((x, y, w, h))

    # sort the digit locations from left-to-right, then initialize the
    # list of classified digits
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])
    print("bounding_boxes len:", len(bounding_boxes))

    # 裁出字符图片
    characters = []
    ctr = 0
    for bbox in bounding_boxes:
        x, y, w, h = bbox
        char_image = bw_image[y:y + h, x:x + w]
        #char_image = cv2.resize(char_image, (20, 20))
        characters.append(char_image)
        #cv2.imshow("char", char_image)
        # only get the front 4
        ctr = ctr + 1
        if ctr >= 4:
            break
    return characters


def findDigits(knn, roi):
    th = cv2.resize(roi, (20, 20))
    out = th.reshape(-1, 400).astype(np.float32)
    ret, result, neighbours, dist = knn.findNearest(out, k=5)
    return int(result[0][0]), th

knn, train, trainLabel = initKnn()
knn, train, trainLabel = updateKnn(knn, train, trainLabel)

img = cv2.imread('./images/code.png')
cv2.imshow('original', img)


width, height = img.shape[:2]
print("width: %d, height: %d" %(width, height))

# rois, edges = findRoi(img, 20)
rois = extract_chars(gray)
digits = []
for r in rois:
    #x, y, w, h = r
    digit, th = findDigits(knn, r)
    print("digit:", digit)
    # code = pytesseract.image_to_string(th, lang='eng', config='--psm 7')
    # print("code = ", code)
#     digits.append(cv2.resize(th, (20, 20)))
#     cv2.rectangle(img, (x, y), (x + w, y + h), (153, 153, 0), 2)
#     cv2.putText(img, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                 (127, 0, 255), 2)
# newEdges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# newFrame = np.hstack((img, newEdges))
# cv2.imshow('frame', newFrame)

cv2.waitKey()

exit(0)

# ================================================================================
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# width = 426
# height = 480
videoFrame = cv2.VideoWriter('frame.avi',
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
                             (int(width) * 2, int(height)), True)
count = 0
while True:
    ret, frame = cap.read()
    frame = frame[:, :width]
    rois, edges = findRoi(frame, 50)
    digits = []
    for r in rois:
        x, y, w, h = r
        digit, th = findDigit(knn, edges[y:y + h, x:x + w], 50)
        digits.append(cv2.resize(th, (20, 20)))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (153, 153, 0), 2)
        cv2.putText(frame, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (127, 0, 255), 2)
    newEdges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    newFrame = np.hstack((frame, newEdges))
    cv2.imshow('frame', newFrame)
    videoFrame.write(newFrame)
    key = cv2.waitKey(1) & 0xff
    if key == ord(' '):
        break
    elif key == ord('x'):
        Nd = len(digits)
        output = concatenate(digits)
        showDigits = cv2.resize(output, (60, 60 * Nd))
        cv2.imshow('digits', showDigits)
        cv2.imwrite(str(count) + '.png', showDigits)
        count += 1
        if cv2.waitKey(0) & 0xff == ord('e'):
            pass
        print('input the digits(separate by space):')
        numbers = input().split(' ')
        Nn = len(numbers)
        if Nd != Nn:
            print('update KNN fail!')
            continue
        try:
            for i in range(Nn):
                numbers[i] = int(numbers[i])
        except:
            continue
        knn, train, trainLabel = updateKnn(knn, train, trainLabel, output,
                                           numbers)
        print('update KNN, Done!')

print('Numbers of trained images:', len(train))
print('Numbers of trained image labels', len(trainLabel))

cap.release()
cv2.destroyAllWindows()
