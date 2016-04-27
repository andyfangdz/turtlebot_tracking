#!/usr/bin/env python
# from __future__ import print_function
import roslib

roslib.load_manifest('cv_test')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from math import radians
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import video
from utils import mark
from pid import PIDController
from path_test import draw_rect

size_treshold = 4
side_inc = 2
size_maxium = 256
flag = True
MorphOps = False
Channel = False
Realtime = False
Update = False


def abs(n):
    if n > 0:
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
    xx0, yy0, xx1, yy1 = max(0, xx0), max(0, yy0), \
                         min(size_maxium, max(1, xx1)), \
                         min(size_maxium, max(1, yy1))  # Make the square's size at least 1 pixel.
    new_window = (xx0, yy0, xx1, yy1)
    return new_window


class App(object):
    def __init__(self, video_src):
        cv2.namedWindow('camshift')
        cv2.setMouseCallback('camshift', self.on_mouse)
        self.mouse_state = 0
        self.selection = None
        self.drag_start = None
        self.tracking_state = 0
        self.show_backproj = False

    def on_mouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y])
        if event == cv2.EVENT_LBUTTONDOWN:
            # print (x,y)
            self.drag_start = (x, y)
            self.tracking_state = 0
            self.mouse_state = 1
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_state:
                h, w = self.frame.shape[:2]
                xo, yo = self.drag_start
                x0, y0 = np.maximum(0, np.minimum([xo, yo], [x, y]))
                x1, y1 = np.minimum([w, h], np.maximum([xo, yo], [x, y]))
                # print (x0,y0,x1,y1)
                self.selection = None
                if x1 - x0 > 0 and y1 - y0 > 0:
                    self.selection = (x0, y0, x1, y1)
                # print self.selection
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_state = 0
            self.drag_start = None
            flag = False
            if self.selection is not None:
                self.tracking_state = 1

    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count * bin_w, 3), np.uint8)
        for i in range(bin_count):
            h = int(self.hist[i])
            cv2.rectangle(img, (i * bin_w + 2, 255), ((i + 1) * bin_w - 2, 255 - h),
                          (int(180.0 * i / bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist', img)

    def run(self, frame):
        global Update
        global MorphOps
        global Channel
        global Realtime
        y = None
        if True:

            self.frame = frame.copy()
            vis = self.frame.copy()
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            mask = cv2.inRange(hsv, np.array((0., 0., 0.)), np.array((180., 255., 255.)))
            if self.selection:
                x0, y0, x1, y1 = self.selection
                self.track_window = (x0, y0, x1 - x0, y1 - y0)
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]

                if Channel:
                    hist = cv2.calcHist([hsv_roi], [0, 1], mask_roi, [16, 5], [0, 180, 0, 256])
                else:
                    hist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                self.hist = hist.reshape(-1)
                self.show_hist()

                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

            if self.tracking_state == 2:
                if Channel:
                    prob = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
                else:
                    prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask
                term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                self.previous_window = self.track_window
                kernel = np.ones((5, 5), np.uint8)
                if MorphOps:
                    prob = cv2.morphologyEx(prob, cv2.MORPH_OPEN, kernel)
                    prob = cv2.morphologyEx(prob, cv2.MORPH_CLOSE, kernel)
                prob = cv2.GaussianBlur(prob, (5, 5), 0)

                track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)
                if get_window_size(self.track_window) <= size_treshold:
                    self.track_window = get_increased_window(self.previous_window)
                    self.tracking_state = 2
                else:
                    self.tracking_state = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                # print "Target Missing."
                cv2.putText(vis, 'Target Missing', (10, 400), font, 1, (255, 255, 255), 2, 1)

            if self.tracking_state == 1:
                self.selection = None
                if Channel:
                    prob = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
                else:
                    prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask
                term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                self.previous_window = self.track_window
                kernel = np.ones((5, 5), np.uint8)
                if MorphOps:
                    prob = cv2.morphologyEx(prob, cv2.MORPH_OPEN, kernel)
                    prob = cv2.morphologyEx(prob, cv2.MORPH_CLOSE, kernel)
                prob = cv2.GaussianBlur(prob, (5, 5), 0)
                track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)
                if get_window_size(self.track_window) <= size_treshold:
                    self.track_window = get_increased_window(self.previous_window)
                    self.tracking_state = 2
                if self.show_backproj:
                    vis[:] = prob[..., np.newaxis]
                xx0, yy0, xx1, yy1 = self.track_window
                img_roi = self.frame[yy0: yy0 + yy1, xx0: xx0 + xx1]
                cv2.imshow("Tracking Window", img_roi)
                if get_window_size(self.track_window) >= size_treshold and Update:
                    self.bkp = self.hist
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(vis, 'Updating...', (10, 200), font, 1, (255, 255, 255), 2, 1)
                    xx0, yy0, xx1, yy1 = self.track_window
                    xx1 /= 3
                    yy1 /= 3
                    xx0 += xx1
                    yy0 += yy1
                    if xx1 > 0 and yy1 > 0:
                        # print self.track_window
                        hsv_roi = hsv[yy0: yy0 + yy1, xx0: xx0 + xx1]
                        mask_roi = mask[yy0: yy0 + yy1, xx0: xx0 + xx1]
                        cv2.imshow("Tracking Window", hsv_roi)
                        hist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
                        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                        # print cv2.compareHist(hist.reshape(-1), self.bkp, 0)
                        self.hist = hist.reshape(-1)
                    self.show_hist()
                    if not Realtime:
                        Update = not Update
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(vis, str(track_box[0]), (10, 400), font, 1, (255, 255, 255), 2, 1)

                y = track_box[0]
                mark.draw_machine_mark(60, track_box[0], vis)

            if flag:
                height, width, _ = vis.shape
                if y:
		                for i in range(1, 5):
		                    draw_rect(vis, depth=i, angle=(y[0] - width/2) / 10)
                cv2.imshow('camshift', vis)

            ch = 0xFF & cv2.waitKey(50)
            if ch == 27:
                return
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
        return y


class image_converter:
    def __init__(self):
        self.image_pub = rospy.Publisher("image_topic_2", Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image", Image, self.depth_callback)
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=1)
        self.app = App(0)
        self.image_updated = False
        self.depth_updated = False
        self.image = None
        self.depth = None
        self.control = PIDController(Kp=0.1, Ki=0, Kd=0)

    def image_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.image_updated = True
        self.callback_uniform()

    def depth_callback(self, data):
        self.depth = self.bridge.imgmsg_to_cv2(data, "16UC1")
        self.depth_updated = True

    def callback_uniform(self):

        if self.image_updated and self.depth_updated:
            cv_image = self.image
            (rows, cols, channels) = cv_image.shape
            ret = self.app.run(cv_image)
            self.image_updated = False
            self.depth_updated = False
            if ret is not None:
                y, x = ret
                # print y, cols
                dist = self.depth[int(x), int(y)][0]

                turn_cmd = Twist()
                turn_cmd.linear.x = 0.1 if dist > 0 else 0
                diff = y - cols / 2
                output = self.control.output(-diff)
                print(output)
                turn_cmd.angular.z = radians(output)
                self.cmd_vel.publish(turn_cmd)


def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    rospy.Rate(30)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
