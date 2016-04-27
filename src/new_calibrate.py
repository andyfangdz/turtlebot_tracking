#!/usr/bin/env python

import numpy as np
import cv2
import os
import sys, getopt
from glob import glob

img_set = 'frame*.jpg'
img_names = glob(img_set)

chessboard_size = (9, 6)
pattern_points = np.zeros( (np.prod(chessboard_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(chessboard_size).T.reshape(-1, 2)

obj_points = []
img_points = []
h, w = 0, 0
for name in img_names:
    print 'Detecting chessboard on %s...' % name,
    img = cv2.imread(name, 0)
    h, w = img.shape[:2]
    found, corners = cv2.findChessboardCorners(img, chessboard_size)
    if found:
        term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, chessboard_size, corners, found)
        cv2.imshow('Corners', vis)
        cv2.imwrite('proc_'+name,vis)
        cv2.waitKey(2000)
    if not found:
        print 'chessboard not found'
        continue
    img_points.append(corners.reshape(-1, 2))
    obj_points.append(pattern_points)
    
    print 'ok'

rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h))
print "RMS:", rms
print "camera matrix:\n", camera_matrix
print "distortion coefficients: ", dist_coefs.ravel()
img = cv2.imread('1.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(camera_matrix,dist_coefs,(w,h),1,(w,h))
# undistort
dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)