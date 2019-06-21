import cv2
import numpy as np

def to_contours_image(contours, ref_image):
    blank_background = np.zeros_like(ref_image)
    img_contours = cv2.drawContours(blank_background, contours, -1, (255, 255, 255), thickness=2)
    return img_contours

# 读取图片
img = cv2.imread("images/meter4.jpeg")

# 转灰度图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯滤波
gaus = cv2.GaussianBlur(gray, (3,3), 0)
cv2.imshow("gaus", gaus)

# 轮廓提取
edges = cv2.Canny(gaus, 50, 150, apertureSize = 3)
cv2.imshow("edges", edges)


sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
erosion = cv2.erode(edges, sqKernel) # 腐蚀


_, binary = cv2.threshold(erosion, 127, 255, cv2.THRESH_BINARY)  

# 轮廓检测
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

minRect = []
for c in contours:
    # 获得最小的矩形轮廓 可能带旋转角度
    rect = cv2.minAreaRect(c)
    minRect.append(rect)


newImg = img.copy()
for rect in minRect:
    if 1:
        # print(rect)
        # 计算最小区域的坐标
        box = cv2.boxPoints(rect)
        # 坐标规范化为整数
        box = np.int0(box)
        # 画出轮廓
        cv2.drawContours(newImg, [box], 0, (0, 0, 255), 3)

# # 新打开一个图片，我这里这张图片是一张纯白图片
# newImg = img.copy()
# #newImg = cv2.resize(newImg, (300,300))

# # 画图
# cv2.drawContours(newImg, contours, -1, (0, 0, 0), 3)
#newImg =  to_contours_image(minRect, img)

# 展示
cv2.imshow("newimg", newImg)  
cv2.waitKey(0) 
