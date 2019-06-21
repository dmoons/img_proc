import cv2
import numpy as np

img1 = cv2.imread('./images/6789-30.png',0)

ret, thresh = cv2.threshold(img1, 127, 255, 0)

contours,hierarchy = cv2.findContours(thresh,2,1)
cnt1 = contours[0]

ret = cv2.matchShapes(cnt1,1,0.0)
print ret