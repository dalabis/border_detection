import FindRectangleHough

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

data = []
with open("input/NN_output/loose") as f:
    for line in f:
        data.append([float(x) for x in line.split()])

ind_num = 3

image = np.array(data)
image_gray = (image / (ind_num - 1) * 255).astype(np.uint8)

height, width = image_gray.shape[:2]
res = cv.resize(image_gray, (2*width, 2*height), interpolation=cv.INTER_AREA)
cv.imshow('image_1', res)
cv.waitKey(0)
cv.destroyAllWindows()

image_ind_1 = (image == 1)
image_ind_1_gray = (image_ind_1 * 255).astype(np.uint8)

height, width = image_ind_1_gray.shape[:2]
res = cv.resize(image_ind_1_gray, (2*width, 2*height),
                interpolation=cv.INTER_AREA)
cv.imshow('image_2', res)
cv.waitKey(0)
cv.destroyAllWindows()

edges_ind_1 = cv.Canny(image_ind_1_gray, 100, 200)

height, width = edges_ind_1.shape[:2]
res = cv.resize(edges_ind_1, (2*width, 2*height), interpolation=cv.INTER_AREA)
cv.imshow('image_3', res)
cv.waitKey(0)
cv.destroyAllWindows()

_, contours_ind_1, _ = cv.findContours(
    image_ind_1_gray, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)


def rectangle_plot(rect):
    plt.plot([rect[0, 0], rect[0, 1], rect[0, 2], rect[0, 3], rect[0, 0]],
        [rect[1, 0], rect[1, 1], rect[1, 2], rect[1, 3], rect[1, 0]])


plt.figure(figsize=(5, 5))
point_cloud = []
for contour in contours_ind_1:
    points = []
    for i in contour:
        points.append(i[0].tolist())
    plt.plot(np.array(points)[:, 0], np.array(points)[:, 1], '.')
    point_cloud.append(np.array(points))

for points in point_cloud:
    lines = FindRectangleHough.find_polygon(points)

    for i in range(lines.shape[0]):

        plt.plot([lines[i, 0], lines[i, 2]], [lines[i, 1], lines[i, 3]], 'k')






height, width = image_gray.shape[:2]
res = cv.resize(image_gray, (2*width, 2*height), interpolation=cv.INTER_AREA)

image_ind_2 = (image == 2)
image_ind_2_gray = (image_ind_2 * 255).astype(np.uint8)

height, width = image_ind_2_gray.shape[:2]
res = cv.resize(image_ind_2_gray, (2*width, 2*height), interpolation=cv.INTER_AREA)

edges_ind_2 = cv.Canny(image_ind_2_gray, 100, 200)

height, width = edges_ind_2.shape[:2]
res = cv.resize(edges_ind_2, (2*width, 2*height), interpolation=cv.INTER_AREA)

_, contours_ind_2, _ = cv.findContours(
    image_ind_2_gray, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)


def rectangle_plot(rect):
    plt.plot([rect[0, 0], rect[0, 1], rect[0, 2], rect[0, 3], rect[0, 0]],
        [rect[1, 0], rect[1, 1], rect[1, 2], rect[1, 3], rect[1, 0]])

point_cloud = []
for contour in contours_ind_2:
    points = []
    for i in contour:
        points.append(i[0].tolist())
    plt.plot(np.array(points)[:, 0], np.array(points)[:, 1], '.')

    point_cloud.append(np.array(points))

rect_points = []
for points in point_cloud:
    rect = FindRectangleHough.find_rectangle(points)
    rectangle_plot(rect)
    rect_points.append(rect)

plt.xlim((-10, 130))
plt.ylim((70, 210))
plt.show()
