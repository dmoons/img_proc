# -*-encoding:utf-8-*-
# pip3 install pytesseract
# pip3 install PILLOW

import pytesseract
from PIL import Image
import cv2
import numpy as np


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
        (x, y, w, h) = x - 1, y - 1, w + 1, h + 1
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
        char_image = bwimg[y:y + h, x:x + w]
        #char_image = cv2.resize(char_image, (20, 20))
        characters.append(char_image)
        cv2.imshow("char" + str(ctr), char_image)
        # only get the front 4
        ctr = ctr + 1
        if ctr >= 4:
            break
    return characters

###########################################################################

#image = Image.open("images/code.png")
# image = image.convert('L')
# image.show()
# code = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
# print("code = ", code)

image = cv2.imread('./images/code.png')
# img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ret, bwimg = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("inv", bwimg)
# #code = pytesseract.image_to_string(bwimg, lang='eng', config='--psm 7')
# code = pytesseract.image_to_string(bwimg, lang="eng", config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789")
# print("code = ", code)
# cv2.waitKey()
rois = extract_chars(image)
digits = []
for r in rois:
    code = pytesseract.image_to_string(r, lang='eng', config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789")
    print("code = ", code)

cv2.waitKey()
