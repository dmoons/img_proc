import os
import cv2
import math
import random
import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt
filepath = './'
 
img = cv2.imread('./images/meter4.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)


cv2.imshow("img", img)
cv2.imshow("gray", gray)
cv2.imshow("edges", edges)


# 霍夫变换
lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
rotate_angle = 0
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    if x1 == x2 or y1 == y2:
        continue
    t = float(y2 - y1) / (x2 - x1)
    rotate_angle = math.degrees(math.atan(t))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
print("rotate_angle : "+str(rotate_angle))
rotate_img = ndimage.rotate(img, rotate_angle)
#misc.imsave('ssss.png',rotate_img)
cv2.imshow("rotate", rotate_img)
cv2.waitKey(0)

# --------------------- 
# 作者：sun_tao1 
# 来源：CSDN 
# 原文：https://blog.csdn.net/sun_tao1/article/details/83344216 
# 版权声明：本文为博主原创文章，转载请附上博文链接！
