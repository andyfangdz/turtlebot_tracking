import numpy as np
import cv2
import math

def match(template, image):
    '''
    Match template on image and return the centroid's coord.
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray, patch, cv2.TM_CCOEFF_NORMED)
    result = np.abs(result) ** 3
    val, result = cv2.threshold(result, 0.01, 0, cv2.THRESH_TOZERO)

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

    height, width, depth = template.shape
    x, y = maxLoc
    x += width/2
    y += height/2
    
    return (x,y)