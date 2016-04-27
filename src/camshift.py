#!/usr/bin/env python

import numpy as np
import cv2
import video
from utils import mark

size_treshold = 4
side_inc = 2
size_maxium = 256
flag= True
MorphOps = False
Channel = False
Realtime = False
Update = False

def abs(n):
    if n>0:
        return n
    return -n

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

class App(object):
    def __init__(self, video_src):
        self.cam = video.create_capture(0)
        ret, self.frame = self.cam.read()
        cv2.namedWindow('camshift')
        cv2.setMouseCallback('camshift', self.onmouse)
        self.mouse_state = 0
        self.selection = None
        self.drag_start = None
        self.tracking_state = 0
        self.show_backproj = False

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
        cv2.imshow('hist', img)

    def run(self):
        global Update
        global MorphOps
        global Channel
        global Realtime
        while True:

            ret, self.frame = self.cam.read()
            vis = self.frame.copy()
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            mask = cv2.inRange(hsv, np.array((0., 0., 0.)), np.array((180., 255., 255.)))
            if self.selection:
                x0, y0, x1, y1 = self.selection
                self.track_window = (x0, y0, x1-x0, y1-y0)
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]
                
                if Channel:
                    hist = cv2.calcHist( [hsv_roi], [0,1], mask_roi, [16,5], [0, 180, 0 ,256] )
                else:
                    hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                self.hist = hist.reshape(-1)
                self.show_hist()

                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

            if self.tracking_state == 2:
                if Channel:
                    prob = cv2.calcBackProject([hsv], [0,1], self.hist, [0, 180, 0, 256], 1)
                else:
                    prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask
                term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
                self.previous_window = self.track_window
                kernel = np.ones((5,5),np.uint8)
                if MorphOps:
                    prob = cv2.morphologyEx(prob, cv2.MORPH_OPEN, kernel)
                    prob = cv2.morphologyEx(prob, cv2.MORPH_CLOSE, kernel)
                prob = cv2.GaussianBlur(prob,(5,5),0)

                track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)
                if get_window_size(self.track_window) <= size_treshold:
                    self.track_window = get_increased_window(self.previous_window)
                    self.tracking_state = 2
                else :
                    self.tracking_state = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                print "Target Missing."
                cv2.putText(vis,'Target Missing',(10,400), font, 1,(255,255,255),2,1)

            if self.tracking_state == 1:
                self.selection = None
                if Channel:
                    prob = cv2.calcBackProject([hsv], [0,1], self.hist, [0, 180, 0, 256], 1)
                else:
                    prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask
                term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
                self.previous_window = self.track_window
                kernel = np.ones((5,5),np.uint8)
                if MorphOps:
                    prob = cv2.morphologyEx(prob, cv2.MORPH_OPEN, kernel)
                    prob = cv2.morphologyEx(prob, cv2.MORPH_CLOSE, kernel)
                prob = cv2.GaussianBlur(prob,(5,5),0)
                track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)
                if get_window_size(self.track_window) <= size_treshold:
                    self.track_window = get_increased_window(self.previous_window)
                    self.tracking_state = 2
                if self.show_backproj:
                    vis[:] = prob[...,np.newaxis]
                xx0, yy0, xx1, yy1 = self.track_window
                img_roi = self.frame[yy0 : yy0 + yy1, xx0 : xx0 + xx1]
                cv2.imshow("Tracking Window",img_roi)
                if get_window_size(self.track_window) >= size_treshold and Update:
                    self.bkp=self.hist
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(vis,'Updating...',(10,200), font, 1,(255,255,255),2,1)
                    xx0, yy0, xx1, yy1 = self.track_window
                    xx1 /= 3
                    yy1 /= 3
                    xx0 += xx1
                    yy0 += yy1
                    if xx1 > 0 and yy1 > 0:
                        print self.track_window
                        hsv_roi = hsv[yy0 : yy0 + yy1, xx0 : xx0 + xx1]
                        mask_roi = mask[yy0 : yy0 + yy1 , xx0 : xx0 + xx1]
                        cv2.imshow("Tracking Window",hsv_roi)
                        hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                        print cv2.compareHist(hist.reshape(-1), self.bkp, 0)
                        self.hist = hist.reshape(-1)
                    self.show_hist()
                    if not Realtime:
                        Update = not Update
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(vis,str(track_box[0]),(10,400), font, 1,(255,255,255),2,1)
                print str(track_box[0])
                #try: cv2.ellipse(vis, track_box, (0, 0, 255), 2)
                #except: print track_box
                mark.draw_machine_mark(60, track_box[0], vis)

            #cv2.imshow('Original Footage',self.frame)
            if flag:
                cv2.imshow('camshift', vis)

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
            if ch == ord('m'):
                MorphOps = not MorphOps
            if ch == ord('c'):
                Channel = not Channel
            if ch == ord('u'):
                Update = not Update
            if ch == ord('r'):
                Realtime = not Realtime
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    try: video_src = sys.argv[1]
    except: video_src = 0
    print __doc__
    App(video_src).run()
