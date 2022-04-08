import cv2
import copy

import numpy as np


def pos2pixel(pos):
    pixel = [0, 0]
    pixel[0] = int(pos[0] * 9.44) + 402   # Left: - || Right: +
    pixel[1] = int(pos[1] * 9.44) - 670   # Up: - || Down: +
    return pixel


def draw_waypoints(image, points):
    img_with_waypoints = image
    point_color = (255, 0, 0)
    for i, point in enumerate(points):
        pixel = pos2pixel(point)
        pixel_text = copy.deepcopy(pixel)
        pixel_text[1] -= 30
        img_with_waypoints = cv2.circle(img_with_waypoints, pixel, 8, point_color, 16)
        # img_with_waypoints = cv2.putText(img_with_waypoints, str(i), pixel_text,
        #                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    return img_with_waypoints


if __name__ == '__main__':
    img_init = cv2.imread('maps/Town02_resize.png')

    vehicle_info = np.load("maps/map2_spawn_points.npy")
    print(vehicle_info.shape)
    vehicle_pos = vehicle_info[:, :2]
    print(vehicle_pos.shape)

    img = copy.deepcopy(img_init)
    img = draw_waypoints(img, vehicle_pos)
    cv2.imwrite("maps/Town02_with_spawn_points.png", img)
