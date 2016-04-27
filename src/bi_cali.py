import numpy as np
import cv2
import os
import sys, getopt
from glob import glob

img_set = '2*.jpg'
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
        cv2.waitKey(0)
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

chessboard_size = (9, 6)
pattern_points = np.zeros( (np.prod(chessboard_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(chessboard_size).T.reshape(-1, 2)

#Define Cameras
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

obj_points = []
img_points = []
img_points2 = []

chessboard_size = (9, 6)
pattern_points = np.zeros( (np.prod(chessboard_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(chessboard_size).T.reshape(-1, 2)

while True:
	ret, img = cap0.read()
	ret, img2 = cap1.read()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	found, corners = cv2.findChessboardCorners(img, chessboard_size)
	found1, corners2 = cv2.findChessboardCorners(img2, chessboard_size)
	if found and found1:
		term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
		cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
		cv2.cornerSubPix(img2, corners2, (5, 5), (-1, -1), term)
		vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		vis2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
		cv2.drawChessboardCorners(vis, chessboard_size, corners, found)
		cv2.drawChessboardCorners(vis2, chessboard_size, corners2, found)
		img=vis
		img2=vis2
		img_points.append(corners.reshape(-1, 2))
		img_points2.append(corners2.reshape(-1, 2))
		obj_points.append(pattern_points)
	cv2.imshow('Image', img)
	cv2.imshow('Image2', img2)
	ch = 0xFF & cv2.waitKey(1)
	if ch == 27:
		break	
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(obj_points,
																							img_points,
																							img_points2,
																							(640,480),
																							cameraMatrix1=camera_matrix,
																							distCoeffs1=dist_coefs,
																							cameraMatrix2=camera_matrix,
																							distCoeffs2=dist_coefs,
																							)
print "-cameraMatrix1:"
print cameraMatrix1
np.save('cameraMatrix1.npy', cameraMatrix1)
print "-distCoeffs1:"
print distCoeffs1
np.save('distCoeffs1.npy', distCoeffs1)
print "-cameraMatrix2:"
print cameraMatrix2
np.save('cameraMatrix2.npy', cameraMatrix2)
print "-distCoeffs2:"
print distCoeffs2
np.save('distCoeffs2.npy', distCoeffs2)
print "-R:"
print R
np.save('R.npy', R)
print "-T:"
print T
np.save('T.npy', T)
print "-E:"
print E
np.save('E.npy', E)
print "-F:"
print F
np.save('F.npy', F)

