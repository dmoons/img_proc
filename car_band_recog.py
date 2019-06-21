# encoding:utf-8
import cv2
import numpy as np

def get_colorvalue(image):
    height, width, shape = image.shape
    image_hsv = np.zeros((height,width), np.uint8)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image_hue, image_saturation, image_value = cv2.split(image_hsv)    
    return image_value
    
def enhance_contrast(image):    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT,kernel)
    img_blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    
    image_plus_tophat = cv2.add(image, img_tophat)
    image_plus_blackhat_minus_blackhat = cv2.subtract(image_plus_tophat, img_blackhat)

    return image_plus_blackhat_minus_blackhat
    
def preprocess(srcimage):
    
    image_value = get_colorvalue(srcimage)    
    image_enhance = enhance_contrast(image_value)
    
    image_blur = cv2.GaussianBlur(image_enhance, (5,5), 0)
    # _, image_binary = cv2.threshold(image_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, image_binary = cv2.threshold(image_blur, 100, 255, cv2.THRESH_BINARY )
    
    cv2.imwrite('image_binary.png',image_binary)
 
    return  image_binary


# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# img_tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT,kernel)
# img_blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)


# image_plus_black = cv2.add(image, img_blackhat)
# image_plus_blackhat_minus_blackhat = cv2.subtract(image_plus_black, img_tophat)

#contants for plate contour
MIN_CONTOUR_WIDTH = 80
MIN_CONTOUR_HEIGHT = 30

MIN_CONTOUR_RATIO = 1.5
MAX_CONTOUR_RATIO = 5

MIN_CONTOURL_AREA = 1500

def get_external_contours(image_thresh):
    #    Construct display images and display contours in images
    
    height, width = image_thresh.shape
    image_contour1 = np.zeros((height, width),np.uint8)
    image_contour2 = np.zeros((height, width),np.uint8) 
    
##    Custom 3*3 nuclei undergo expansion corrosion in the X direction    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image_dilate= cv2.dilate(image_thresh,kernel,iterations =2)
    image_erode= cv2.erode(image_dilate, kernel, iterations = 4)
    image_dilate= cv2.dilate(image_erode,kernel,iterations = 2)
#    
    _, contours, hierarchy = cv2.findContours(image_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_filter = []
    cv2.drawContours(image_contour1, contours, -1,(255, 255, 255 ),3)
        
#    choose the suite contour by the feature of special scence
    for contour in contours:
        contour_possible = PossibleContour(contour)
        if(check_external_contour(contour_possible)): 
            cv2.rectangle(image_contour2, (contour_possible.rectX, contour_possible.rectY),
                                      (contour_possible.rectX + contour_possible.rectWidth, contour_possible.rectY + contour_possible.rectHeight),255)
            contour_filter.append(contour_possible)
            
    print("the length of origin contours is %d " %len(contour_filter))
    cv2.imwrite("1_1contours.png", image_contour1)
    cv2.imwrite("1_2contours.png", image_contour2)
    return contour_filter
    
#    #According to the license plate area size, length and width ratio, the primary screening is carried out
def check_external_contour(contour):
    if (contour.area > MIN_CONTOURL_AREA and contour.rectWidth > MIN_CONTOUR_WIDTH and contour.rectHeight > MIN_CONTOUR_HEIGHT
        and contour.whratio > MIN_CONTOUR_RATIO and contour.whratio < MAX_CONTOUR_RATIO):
        return True
    else:
        return False





