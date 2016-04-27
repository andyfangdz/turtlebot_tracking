#!/usr/bin/env python

import Queue
import threading

import numpy as np
import cv2
import cv2.cv as cv
import video
from utils import mark

READY = [False, False]

def isReady():
    if READY:
        if READY[0] and READY[1]:
            return True
    return False

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

    #x
    cv2.line(traj, project((-imgsize/2, 0, 0), imgsize), project((imgsize/2, 0, 0), imgsize), BLUE, size)
    cv2.line(traj, project((imgsize/2, 0, 0), imgsize), project((imgsize/2 - k, -k, 0), imgsize), BLUE, size)
    cv2.line(traj, project((imgsize/2, 0, 0), imgsize), project((imgsize/2 - k, k, 0), imgsize), BLUE, size)

    #y
    cv2.line(traj, project((0, -imgsize/2, 0), imgsize), project((0, imgsize/2, 0), imgsize), GREEN, size)
    cv2.line(traj, project((0, imgsize/2, 0), imgsize), project((k, imgsize/2 - k, 0), imgsize), GREEN, size)
    cv2.line(traj, project((0, imgsize/2, 0), imgsize), project((-k, imgsize/2 - k, 0), imgsize), GREEN, size)

    #z
    cv2.line(traj, project((0, 0, -imgsize/2), imgsize), project((0, 0, imgsize/2), imgsize), RED, size)
    cv2.line(traj, project((0, 0, imgsize/2), imgsize), project((k, 0, imgsize/2 - k), imgsize), RED, size)
    cv2.line(traj, project((0, 0, imgsize/2), imgsize), project((-k, 0, imgsize/2 - k), imgsize), RED, size)

    #drawArrow(img, size, (imgsize / 2, imgsize / 2), imgsize / 2,  0, BLUE)
    #drawArrow(img, size, (imgsize / 2, imgsize / 2), imgsize * 1.414 / 2, math.pi / 4, GREEN)
    #drawArrow(img, size, (imgsize / 2, imgsize / 2), imgsize / 2,  math.pi / 2, RED)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

points = [(0,0,0)]

def addAnother(pt):
    global points
    points.append(pt)
    cv2.line(traj, project(points[len(points)-2],512),project(pt,512), WHITE, 2)
    cv2.imshow("trajectory", traj)
    print "here am I"
    cv2.waitKey(10)



# Create a black image
traj = np.zeros((512,512,3), np.uint8)
drawFancyAxis(traj, 512, 1)
cv2.imshow("trajectory", traj)



size_treshold = 4
side_inc = 2
size_maxium = 256
flag= True
MorphOps = False
Channel = False
Realtime = False
Update = False
bflag = True
show_backproj = False
status = {}
def abs(n):
    if n>0:
        return n
    return -n
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

def get_window_size(window):
    x0, y0, x1, y1 = window
    size = abs(x1) * abs(y1)
    return size

def get_increased_window(window):
    xx0, yy0, xx1, yy1 = window
    xx0 -= side_inc
    yy0 -= side_inc
    xx1 += side_inc
    yy1 += side_inc
    xx0, yy0, xx1, yy1 = max(0, xx0), max(0, yy0), min(size_maxium, max(1, xx1)), min(size_maxium, max(1, yy1))# Make the square's size at least 1 pixel.
    new_window = (xx0, yy0, xx1, yy1)
    return new_window

class Tracker(object):
    def __init__(self, video_src):
        self.cam = video.create_capture(video_src)
        ret, self.frame = self.cam.read()
        cv2.namedWindow('camshift'+str(video_src))
        cv2.setMouseCallback('camshift'+str(video_src), self.onmouse)
        self.mouse_state = 0
        self.selection = None
        self.drag_start = None
        self.tracking_state = 0
        self.show_backproj = False
        self.video_src=video_src
        
    def onmouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y]) # BUG
        #print (x,y)
        debug={}
        bkp=flags
        if event == cv2.EVENT_LBUTTONDOWN:
            #print (x,y)
            self.drag_start = (x, y)
            self.tracking_state = 0
            self.mouse_state = 1
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_state:
                h, w = self.frame.shape[:2]
                xo, yo = self.drag_start
                x0, y0 = np.maximum(0, np.minimum([xo, yo], [x, y]))
                x1, y1 = np.minimum([w, h], np.maximum([xo, yo], [x, y]))
                #print (x0,y0,x1,y1)
                self.selection = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.selection = (x0, y0, x1, y1)
                    #print self.selection
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_state = 0
            self.drag_start = None
            flag= False
            if self.selection is not None:
                self.tracking_state = 1 

    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist'+str(self.video_src), img)

    def make_selection(self):
        x0, y0, x1, y1 = self.selection
        self.track_window = (x0, y0, x1-x0, y1-y0)
        self.hsv_roi = self.hsv[y0:y1, x0:x1]
        self.mask_roi = self.mask[y0:y1, x0:x1]
        if Channel:
            self.hist = cv2.calcHist( [self.hsv_roi], [0,1], self.mask_roi, [16,5], [0, 180, 0 ,256] )
        else:
            self.hist = cv2.calcHist( [self.hsv_roi], [0], self.mask_roi, [16], [0, 180] )
        cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
        self.hist = self.hist.reshape(-1)
        self.show_hist()

        self.vis_roi = self.vis[y0:y1, x0:x1]
        cv2.bitwise_not(self.vis_roi, self.vis_roi)
        self.vis[self.mask == 0] = 0

    def get_img(self):
        self.ret, self.frame = self.cam.read()
        self.vis = self.frame.copy()
        self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(self.hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        

    def track(self):
        if Channel:
            self.prob = cv2.calcBackProject([self.hsv], [0,1], self.hist, [0, 180, 0, 256], 1)
        else:
            self.prob = cv2.calcBackProject([self.hsv], [0], self.hist, [0, 180], 1)
        self.prob &= self.mask
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        self.previous_window = self.track_window
        kernel = np.ones((5,5),np.uint8)
        if MorphOps:
            self.prob = cv2.morphologyEx(prob, cv2.MORPH_OPEN, kernel)
            self.prob = cv2.morphologyEx(prob, cv2.MORPH_CLOSE, kernel)
        self.prob = cv2.GaussianBlur(self.prob,(5,5),0)
        self.track_box, self.track_window = cv2.CamShift(self.prob, self.track_window, term_crit)
        if get_window_size(self.track_window) <= size_treshold:
            self.track_window = get_increased_window(self.previous_window)
            self.tracking_state = 2
        else :
            self.tracking_state = 1

    def run(self):
        global Update
        global MorphOps
        global Channel
        global Realtime
        global bflag
        global show_backproj
        
        self.get_img()
        
        if self.selection:
            
            self.make_selection()
        if self.tracking_state == 2:
            READY[self.video_src] = False
            self.track()
            font = cv2.FONT_HERSHEY_SIMPLEX
            print "Target Missing."
            cv2.putText(self.vis,'Target Missing',(10,400), font, 1,(255,255,255),2,1)
        elif self.tracking_state == 1:
            READY[self.video_src] = True
            self.selection = None
            self.track()
            if self.show_backproj:
                self.vis[:] = prob[...,np.newaxis]
            xx0, yy0, xx1, yy1 = self.track_window
            img_roi = self.frame[yy0 : yy0 + yy1, xx0 : xx0 + xx1]
            cv2.imshow("Tracking Window"+str(self.video_src),img_roi)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.vis,str(self.track_box[0]),(10,400), font, 1,(255,255,255),2,1)
            status[self.video_src]=self.track_box[0]
            #print str(track_box[0])
            #try: cv2.ellipse(vis, track_box, (0, 0, 255), 2)
            #except: print track_box
            mark.draw_machine_mark(60, self.track_box[0], self.vis)

        #cv2.imshow('Original Footage',self.frame)
        if flag:
            cv2.imshow('camshift'+str(self.video_src), self.vis)

        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            bflag = False
        if ch == ord('b'):
            show_backproj = not show_backproj
        if ch == ord('m'):
            MorphOps = not MorphOps
        if ch == ord('c'):
            Channel = not Channel
        if ch == ord('u'):
            Update = not Update
        if ch == ord('r'):
            Realtime = not Realtime
    cv2.destroyAllWindows()
def add(pt1,pt2):
    (x1,y1,z1)=pt1
    (x2,y2,z2)=pt2
    return (x1+x2,y1+y2,z1+z2)
def aver(pt):
    (x,y,z)=pt
    return (x*50,y*50,z*50)

def trans(pt):
    (x,y,z)=pt
    return (x*10,y*10,0)

if __name__ == '__main__':
    import sys
    tick = 0

    temp = (0,0,0)
    a=Tracker(0)
    b=Tracker(1)
    tgt = temp
    last = temp
    while True and bflag:
        status[0]=(0,0)
        status[1]=(0,0)
        a.run()
        b.run()
        #print status
        
        if isReady():
            x0, y0= status[0]
            x1, y1= status[1]
            d = x0 - x1;

            X = x0 * Q[0, 0] + Q[0, 3];
            Y = y0 * Q[1, 1] + Q[1, 3];
            Z = Q[2, 3];
            W = d * Q[3, 2] + Q[3, 3];

            X = X / W;
            Y = Y / W;
            Z = Z / W;

            (tX,tY,tZ) = temp
            (tgX,tgY,tgZ) = tgt
            (laX,laY,laZ) = last
            temp = ((tX+X)/2,(tY+Y)/2,(tZ+Z)/2)
            smo = ((tick%5)*(tgX+laX)/5,(tick%5)*(tgY+laY)/5,(tick%5)*(tgZ+laZ)/5)
            try:
                addAnother(trans(temp))
            except OverflowError:
                pass
            tick += 1
            if tick % 5 ==0:
                last = tgt
                tgt = temp
                

    
