#generate.py：分离整张字符图像代码
# 将字符图片分离成字符
import os
import cv2
import numpy as np

# 提取字符函数
def extract_chars(img):
    # 使用二值化处理
    ret, bwimg = cv2.threshold(img, 64, 255, cv2.THRESH_BINARY)

    # 定义一个3*3的十字结构
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 对图像进行腐蚀,缩小边缘
    erode = cv2.erode(bwimg, kernel, iterations = 1)

    # 图像非运算，黑白颠倒
    bw_image = cv2.bitwise_not(erode)
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)

    # 检测图像轮廓，返回轮廓属性
    contours = cv2.findContours(bw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    # 生成轮廓框
    bounding_boxes = []
    for contour in contours:
        # 轮廓基点和长宽
        x,y,w,h = cv2.boundingRect(contour)
        x,y,w,h = x-2, y-2, w+4, h+4
        bounding_boxes.append((x,y,w,h))

    # 裁出字符图片
    characters = []
    for bbox in bounding_boxes:
        x,y,w,h = bbox
        char_image = img[y:y+h,x:x+w]
        characters.append(char_image)
    return characters

# 存储字符函数
def output_chars(chars, labels):
    for i, char in enumerate(chars):
        # cv2.imshow("Image", char)
        # cv2.waitKey(0)
        filename = "chars/%s.png" % labels[i]
        # 图像长宽放大三倍，邻域双三次插值
        char = cv2.resize(char
            , None
            , fx=3
            , fy=3
            , interpolation=cv2.INTER_CUBIC)
        # char = cv2.resize(char,(90,140))
        cv2.imwrite(filename, char)

# 生成文件夹
if not os.path.exists("chars"):
    os.makedirs("chars")

# 读取灰度数字和字母图片
img_digits = cv2.imread("ch/digit.jpg", 0)
# img_letters = cv2.imread("ch/letter.jpg", 0)

# 提取字符
digits = extract_chars(img_digits)

#letters = extract_chars(img_letters)
DIGITS = [1, 0, 9, 8 ,7, 6, 5, 4, 3, 2]
# LETTERS = [chr(ord('A') + i) for i in range(25,-1,-1)]
#LETTERS = ['N', 'M', 'L', 'K' ,'J', 'I', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A', 'Z', 'Y', 'X', 'W', 'V', 'U', 'T', 'S', 'R', 'Q', 'P', 'O']
# 以字符命名方式存储字符图片
output_chars(digits, DIGITS)
#output_chars(letters, LETTERS) 

