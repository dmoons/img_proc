import cv2
import numpy as np

# Step1. 转换为HSV
img = cv2.imread('images/meter4.jpeg')
hue_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Step2. 用颜色分割图像
#low_range = np.array([0, 123, 100])
low_range = np.array([0, 12, 10])
high_range = np.array([5, 255, 255])
th = cv2.inRange(hue_image, low_range, high_range)

# Step3. 形态学运算，膨胀
dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2)
cv2.imshow("dilate", dilated)

# Step4. Hough Circle
circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 1, 100, param1=15, param2=7, minRadius=2, maxRadius=100)
print("total find:", len(circles))

# Step5. 绘制
if circles is not None:
    for j in range(0, len(circles)):
        print("circles at", j, circles[j])
        x, y, radius = circles[j][0]
        center = (x, y)
        img = cv2.circle(img, center, radius, (0, 255, 0), 2)

cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------- 
# 作者：两鬓已不能斑白 
# 来源：CSDN 
# 原文：https://blog.csdn.net/u010429424/article/details/72989870 
# 版权声明：本文为博主原创文章，转载请附上博文链接！