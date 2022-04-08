import cv2
import numpy as np


def pixel2pos(pixel):
    pos = [0, 0]
    pos[0] = (pixel[0]*2.5 - 402) / 9.44
    pos[1] = (pixel[1]*2.5 + 670) / 9.44
    return pos


def on_button_down(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Pixel:", (y, x), "Carla Pos", pixel2pos([x, y]))


if __name__ == '__main__':
    img = cv2.imread('maps/Town02_with_spawn_points.png')
    img = cv2.resize(img, (1000, 1000))
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_button_down)
    cv2.imshow("image", img)
    cv2.waitKey(0)
