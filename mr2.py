import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import seaborn as sns
import cv2

im = mpimg.imread("./2018shuibiao.jpeg")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   #转换了灰度化
plt.figure('picture')
plt.imshow(im_gray, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# 灰度直方图
plt.figure('displot')
sns.distplot(im_gray.flatten(), kde=False, color='b')
plt.show()


# Canny边缘检测
gaus = cv2.GaussianBlur(gray,(3,3),0)

#主要调整这个值来决定Canny检测的精密度
low_thre = 110
#阈值上界定为1.2倍
edges = cv2.Canny(gaus, low_thre, low_thre*1.2, apertureSize=3)

plt.figure('Edge')
plt.imshow(edges, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# HoughLinesP识别方框

minLineLength = 35
# 同一方向上两条线段判定为一条线段的最大允许间隔（断裂），超过了设定值，则把两条线段当成一条线段，值越大，允许线段上的断裂越大，越有可能检出潜在的直线段
maxLineGap = 10
# 超过设定阈值才被检测出线段，值越大，基本上意味着检出的线段越长，检出的线段个数越少。
threshold = 55

lines = cv2.HoughLinesP(edges, 2.0, np.pi / 180, threshold, maxLineGap=2)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

#cv2.imshow("houghline",img)
cv2.imwrite('shuibiao_test.jpg', img)
cv2.waitKey()
cv2.destroyAllWindows()

# x1 = 97
# x2 = 133
# y1 = 114
# y2 = 236
# rectangle = img[x1:x2, y1:y2]
# cv2.imwrite('shuibiao_rectangle.jpg', rectangle)
