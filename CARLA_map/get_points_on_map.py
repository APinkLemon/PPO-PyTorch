import cv2
import numpy as np


def on_button_down(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(y, x)


# 图片路径
img = cv2.imread('test.png')

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_button_down)
cv2.imshow("image", img)
cv2.waitKey(0)
