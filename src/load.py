import numpy as np
import cv2
import cv2.cv as cv

cameraMatrix1=np.load('cameraMatrix1.npy')
print "-cameraMatrix1:"
print cameraMatrix1
distCoeffs1=np.load('distCoeffs1.npy')
print "-distCoeffs1:"
print distCoeffs1
cameraMatrix2=np.load('cameraMatrix2.npy')
print "-cameraMatrix2:"
print cameraMatrix2
distCoeffs2=np.load('distCoeffs2.npy')
print "-distCoeffs2:"
print distCoeffs2
R=np.load('R.npy')
print "-R:"
print R
T=np.load('T.npy')
print "-T:"
print T
E=np.load('E.npy')
print "-E:"
print E
F=np.load('F.npy' )
print "-F:"
print F
R1=np.array((3,3))
R2=np.array((3,3))
P1=np.array((3,4))
P2=np.array((3,4))
Q=np.array((4,4))
cameraMatrix1=cv.fromarray(cameraMatrix1)
cameraMatrix2=cv.fromarray(cameraMatrix2)
distCoeffs1=cv.fromarray(distCoeffs1)
distCoeffs2=cv.fromarray(distCoeffs2)
R=cv.fromarray(R)
T=cv.fromarray(T)
R1=cv.CreateMat(3, 3, cv.CV_64FC1)
R2=cv.CreateMat(3, 3, cv.CV_64FC1)
P1=cv.CreateMat(3, 4, cv.CV_64FC1)
P2=cv.CreateMat(3, 4, cv.CV_64FC1)
Q=cv.CreateMat(4, 4, cv.CV_64FC1)
cv.StereoRectify(cameraMatrix1,  cameraMatrix2,distCoeffs1, distCoeffs2, (640,480), R, T, R1, R2, P1, P2, Q=Q) 
Q = np.asarray(Q)
print '-Q:'
print Q