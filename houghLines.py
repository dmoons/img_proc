import cv2
import numpy as np
import matplotlib.pyplot as plt

def tophat(gray):
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

    return thresh

#=======================================================================
img = cv2.imread("./images/meter4.jpeg")
# img = cv2.imread("./images/6789-30.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gaus = cv2.GaussianBlur(gray, (3,3), 0)
cv2.imshow("gaus", gaus)

# Step3. 形态学运算，膨胀
dilated = cv2.dilate(gaus, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
cv2.imshow("dilate", dilated)

# thresh1 = tophat(dilated)
# cv2.imshow("thresh1", thresh1)

edges = cv2.Canny(dilated, 10, 155, apertureSize = 3)
cv2.imshow("edges", edges)

# sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# erosion = cv2.erode(edges, sqKernel) # 腐蚀
# cv2.imshow("erosion", erosion)


ret, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh", thresh)
cv2.waitKey(0)

result = img.copy()  
 
#经验参数  
minLineLength = 100
maxLineGap = 100
lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 80, minLineLength, maxLineGap)
for j in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[j]:  
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0), 2)  
  
cv2.imshow('Result', img)  

cv2.waitKey(0)  
cv2.destroyAllWindows()