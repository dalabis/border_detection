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

image_ind_2 = (image == 2)
image_ind_2_gray = (image_ind_2 * 255).astype(np.uint8)

height, width = image_ind_2_gray.shape[:2]
res = cv.resize(image_ind_2_gray, (2*width, 2*height), interpolation=cv.INTER_AREA)
cv.imshow('image_2', res)
cv.waitKey(0)
cv.destroyAllWindows()

edges_ind_2 = cv.Canny(image_ind_2_gray, 100, 200)

height, width = edges_ind_2.shape[:2]
res = cv.resize(edges_ind_2, (2*width, 2*height), interpolation=cv.INTER_AREA)
cv.imshow('image_3', res)
cv.waitKey(0)
cv.destroyAllWindows()

_, contours_ind_2, _ = cv.findContours(
    image_ind_2_gray, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)


def rectangle_plot(rect):
    plt.plot([rect[0, 0], rect[0, 1], rect[0, 2], rect[0, 3], rect[0, 0]],
        [rect[1, 0], rect[1, 1], rect[1, 2], rect[1, 3], rect[1, 0]])


plt.figure(figsize=(5, 5))
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

plt.show()

X = []
Y = []
for rect in rect_points:
    X.append((rect[0, 1] - rect[0, 0]) * (rect[0, 2] - rect[0, 1]))
    X.append((rect[0, 2] - rect[0, 1]) * (rect[0, 3] - rect[0, 2]))
    X.append((rect[0, 3] - rect[0, 2]) * (rect[0, 0] - rect[0, 3]))
    X.append((rect[0, 0] - rect[0, 3]) * (rect[0, 1] - rect[0, 0]))

    Y.append((rect[1, 1] - rect[1, 0]) * (rect[1, 2] - rect[1, 1]))
    Y.append((rect[1, 2] - rect[1, 1]) * (rect[1, 3] - rect[1, 2]))
    Y.append((rect[1, 3] - rect[1, 2]) * (rect[1, 0] - rect[1, 3]))
    Y.append((rect[1, 0] - rect[1, 3]) * (rect[1, 1] - rect[1, 0]))

# LeastSqr
A = np.array(X)
b = - np.array(Y)
x = np.dot(1 / (np.dot(np.transpose(A), A)), np.dot(np.transpose(A), b))

# Коэффициент на который нужно домножать x кординату
alpha = np.sqrt(x)
print('alpha = ' + str(alpha))

plt.figure(figsize=(5, 5))
point_cloud = []
for contour in contours_ind_2:
    points = []
    for i in contour:
        points.append(i[0].tolist())
    plt.plot(alpha * np.array(points)[:, 0], np.array(points)[:, 1], '.')

    point_cloud.append(np.array(points))

rect_points = []
for points in point_cloud:
    rect = FindRectangleHough.find_rectangle(points)
    rect[0, :] = alpha * rect[0, :]
    rectangle_plot(rect)
    rect_points.append(rect)

plt.show()

height, width = image_gray.shape[:2]
res = cv.resize(image_gray, (int(2*width*alpha), 2*height),
                interpolation=cv.INTER_AREA)
cv.imshow('image_1', res)
cv.waitKey(0)
cv.destroyAllWindows()
