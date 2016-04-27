from __future__ import division
import cv2
import numpy as np
import math


def draw_rect(canvas, depth=0.5, angle=-40, margin=0.1):
    height, width, _ = canvas.shape
    margin *= depth
    margin_w, margin_h = width * margin, height * margin
    processed_val_padding = math.sin(angle / 90) * 0.3
    processed_val_margin = math.sin(angle / 90) * 1
    padding_left = margin_h * max(1, (1 + processed_val_padding))
    padding_right = margin_h * max(1, (1 - processed_val_padding))
    margin_left = margin_w * (1 + processed_val_margin)
    margin_right = margin_w * (1 - processed_val_margin)

    points = np.array([[margin_left, padding_left],
                       [width - margin_right, padding_right],
                       [width - margin_right, height - padding_right],
                       [margin_left, height - padding_left]], dtype='int32')
    cv2.polylines(canvas, [points], True, (0, 255, 0))


def main():
    cv2.namedWindow("Canvas")
    canvas = np.zeros((480, 640, 3), np.uint8)
    cv2.imshow("Canvas", canvas)
    cv2.waitKey(2000)
    incremental = 1
    while cv2.waitKey(16) != ' ':
        incremental += 1
        incremental %= 100
        canvas = np.zeros((480, 640, 3), np.uint8)
        for i in range(1, 5):
            draw_rect(canvas, depth=i - incremental / 100)
            cv2.imshow("Canvas", canvas)
            # cv2.waitKey(100)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
