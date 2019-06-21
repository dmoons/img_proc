import cv2
import numpy as np

img = cv2.imread("./images/meter4.jpeg") 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
#Get edges
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv2.imshow("tophat", tophat)

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
cv2.imshow('thresh 1', thresh)

# apply a second closing operation to the binary image, again
# to help close gaps between credit card number regions
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv2.imshow('thresh 2', thresh)


cv2.imshow("Image", img)

 
cv2.waitKey(0)