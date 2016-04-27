import numpy as np
import cv2
import math
import cv2.cv as cv
from itertools import tee, izip

WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

def project(pt, imgsize):
    center = imgsize / 2
    (x, y, z) = pt
    return (int(z * 1.414 / 2 + x + center), int (center -(z * 1.414 /2 +y)))

def drawFancyAxis(img, imgsize, size):
    '''
    Draw a fancy axis with x - blue, y - red, z- green)
    '''
    k = 20

    #x with Arrow
    cv2.line(img, project((-imgsize/2, 0, 0), imgsize), project((imgsize/2, 0, 0), imgsize), BLUE, size)
    cv2.line(img, project((imgsize/2, 0, 0), imgsize), project((imgsize/2 - k, -k, 0), imgsize), BLUE, size)
    cv2.line(img, project((imgsize/2, 0, 0), imgsize), project((imgsize/2 - k, k, 0), imgsize), BLUE, size)

    #y with Arrow
    cv2.line(img, project((0, -imgsize/2, 0), imgsize), project((0, imgsize/2, 0), imgsize), GREEN, size)
    cv2.line(img, project((0, imgsize/2, 0), imgsize), project((k, imgsize/2 - k, 0), imgsize), GREEN, size)
    cv2.line(img, project((0, imgsize/2, 0), imgsize), project((-k, imgsize/2 - k, 0), imgsize), GREEN, size)

    #z with Arrow
    cv2.line(img, project((0, 0, -imgsize/2), imgsize), project((0, 0, imgsize/2), imgsize), RED, size)
    cv2.line(img, project((0, 0, imgsize/2), imgsize), project((k, 0, imgsize/2 - k), imgsize), RED, size)
    cv2.line(img, project((0, 0, imgsize/2), imgsize), project((-k, 0, imgsize/2 - k), imgsize), RED, size)

    #drawArrow(img, size, (imgsize / 2, imgsize / 2), imgsize / 2,  0, BLUE)
    #drawArrow(img, size, (imgsize / 2, imgsize / 2), imgsize * 1.414 / 2, math.pi / 4, GREEN)
    #drawArrow(img, size, (imgsize / 2, imgsize / 2), imgsize / 2,  math.pi / 2, RED)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)



# Create a black image
img = np.zeros((512,512,3), np.uint8)

points = [(-100,-100,-100)]

for tick in range(20):
    (x,y,z) = points[len(points)-1]
    if tick % 6 == 0:
        x += 40
    elif tick % 6 ==1:
        y += 70
    elif tick % 6 ==2:
        z += 60
    elif tick % 6 ==3:
        y -= 50
    elif tick % 6 ==4:
        x -= 70
    elif tick % 6 ==5:
        z -= 40
    points.append((x,y,z))

drawFancyAxis(img, 512, 1)
cv2.imshow("Image", img)
cv2.waitKey(1000)
for _start, _end in pairwise(points):
    cv2.imshow("Image", img)
    cv2.waitKey(150)
    cv2.line(img, project(_start, 512), project(_end, 512), WHITE, 1)

cv2.imshow("Image", img)

cv2.waitKey(0)
