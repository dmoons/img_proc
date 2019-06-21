# -*- coding: utf-8 -*-

# import the necessary packages
from imutils import contours
import numpy as np
from matplotlib import pyplot as plt
import imutils
import cv2
import pytesseract
from PIL import Image
import math

# circle.py


def process_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(0,0),fx=2.0,fy=2.0)
    height, width, depth = img.shape
    print("\n---------------------------------------------\n")
    print("In Process Image Path is %s height is %d Width is %d depth is %d" %(path, height, width, depth))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 15)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 100)

    print("circles count:", len(circles))

    # ensure at least one circles is found, which is our meter
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print("Meter Found")
        for i in circles[0]:
            CenterX = i[0]
            CenterY = i[1]
            Radius = i[2]
            circle_img = np.zeros((height, width), np.uint8)
            cv2.circle(circle_img, (CenterX, CenterY), Radius, 1, thickness=-1)
            masked_data = cv2.bitwise_and(img, img, mask=circle_img)
            output = masked_data.copy()
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
            #cv2.imwrite("output_" + str(index) + ".jpg", output)
            break

        cv2.imshow("output_", output)
        cv2.waitKey(0)

        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray,(5,5),1)
        edged = cv2.Canny(blurred, 5,10,200)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)

        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        displayCnt = None
        contour_list = []

        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c,peri, True)
            # if the contour has four vertices, then we have found
            # the meter display
            if len(approx) == 4:
                contour_list.append(c)
                cv2.contourArea(c)
                displayCnt = approx
                break

        warped = four_point_transform(gray, displayCnt.reshape(4, 2))
        output = four_point_transform(output, displayCnt.reshape(4, 2))
        thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 31, 2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        digitCnts = []

        # loop over the digit area candidates
        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # if the contour is sufficiently large, it must be a digit
            if (w > 5 and w < 100) and (h >= 15 and h <= 150) :
                digitCnts.append(c)

        # sort the contours from left-to-right, then initialize the
        # actual digits themselves
        digitCnts = contours.sort_contours(digitCnts,method="left-to-right")[0]

        mask = np.zeros(thresh.shape, np.uint8)
        cv2.drawContours(mask, digitCnts, -80, (255, 255, 255),-1)
        mask = cv2.bitwise_not(mask)
        mask = cv2.resize(mask, (0, 0), fx=2.0, fy=2.0)
        result = os.popen('/usr/local/bin/ssocr --number-digits=-1 -t 10 Mask.jpg')
        output = result.read()
        print("Output is " + output)
        output = output[2:8]
        return str(round(float(output) * 0.1, 1))
    else:
        print("Circle not Found")
        print("\n---------------------------------------------\n")
        return None


process_image("images/test.jpeg")