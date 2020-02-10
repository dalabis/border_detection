import time
start_time = time.time()

import MyHoughTransform

import numpy as np
from numpy import unravel_index
import cv2 as cv
from matplotlib import pyplot as plt

""" hyperparameters """
delta_rho = 0.5
delta_theta = np.pi / 180
kernel_size = 0.7


def find_two_max(rho_line):
    ind_line_1 = 0
    ind_line_2 = 0
    rho_line_1 = 0
    rho_line_2 = 0
    for i in range(1, rho_line.shape[0] - 1):
        if rho_line[i] > rho_line[i - 1] and \
                rho_line[i] > rho_line[i + 1] and \
                rho_line[i] > rho_line_1:
            temp = ind_line_1
            ind_line_1 = i
            ind_line_2 = temp
            temp = rho_line_1
            rho_line_1 = rho_line[i]
            rho_line_2 = temp
        elif rho_line[i] > rho_line[i - 1] and \
                rho_line[i] > rho_line[i + 1] and \
                rho_line[i] > rho_line_2:
            ind_line_2 = i
            rho_line_2 = rho_line[i]

    return [ind_line_1, ind_line_2]


def find_rectangle_param(points):


    def find_intersections(rho_lines, theta_lines):
        k = -np.cos(theta_lines) / np.sin(theta_lines)
        m = rho_lines / np.sin(theta_lines)

        result_points = np.zeros((2, 4))
        result_points[:, 0] = ([
            - (m[0] - m[1]) / (k[0] - k[1]),
            - k[0] * (m[0] - m[1]) / (k[0] - k[1]) + m[0],
        ])
        result_points[:, 1] = ([
            - (m[1] - m[2]) / (k[1] - k[2]),
            - k[1] * (m[1] - m[2]) / (k[1] - k[2]) + m[1],
        ])
        result_points[:, 2] = ([
            - (m[2] - m[3]) / (k[2] - k[3]),
            - k[2] * (m[2] - m[3]) / (k[2] - k[3]) + m[2],
        ])
        result_points[:, 3] = ([
            - (m[3] - m[0]) / (k[3] - k[0]),
            - k[3] * (m[3] - m[0]) / (k[3] - k[0]) + m[3],
        ])

        return result_points

    acc_array, rho, theta, x_centroid, y_centroid = \
        MyHoughTransform.kernel_hough_tramsform(
            points, delta_rho, delta_theta, kernel_size, 0)

    # acc_gray = (acc_array / np.max(acc_array) * 255).astype(np.uint8)
    # scale = 4
    # height, width = acc_gray.shape[:2]
    # res = cv.resize(acc_gray, (scale * width, scale * height),
    #                 interpolation=cv.INTER_AREA)

    # cv.imshow('image', res)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    theta_long_line_ind = unravel_index(acc_array.argmax(), acc_array.shape)[1]
    if theta_long_line_ind + acc_array.shape[1] // 2 > acc_array.shape[1]:
        theta_short_line_ind = theta_long_line_ind - acc_array.shape[1] // 2
    else:
        theta_short_line_ind = theta_long_line_ind + acc_array.shape[1] // 2

    pho_long_line = acc_array[:, theta_long_line_ind]
    pho_short_line = acc_array[:, theta_short_line_ind]

    # plt.figure()
    # plt.plot(rho, pho_long_line)
    # plt.plot(rho, pho_short_line)
    # plt.show()

    long_lines = find_two_max(pho_long_line)
    short_lines = find_two_max(pho_short_line)

    rho_lines = np.array([rho[long_lines[0]], rho[short_lines[0]],
                          rho[long_lines[1]], rho[short_lines[1]]])
    theta_lines = np.array(
        [theta[theta_long_line_ind], theta[theta_short_line_ind],
         theta[theta_long_line_ind], theta[theta_short_line_ind]])

    result_points = find_intersections(rho_lines, theta_lines)

    result_points[0, :] += x_centroid
    result_points[1, :] += y_centroid

    return result_points, theta_long_line_ind


data = []
with open("input/NN_output/loose") as f:
    for line in f:
        data.append([float(x) for x in line.split()])

ind_num = 3

image = np.array(data)
image_gray = (image / (ind_num - 1) * 255).astype(np.uint8)

height, width = image_gray.shape[:2]
res = cv.resize(image_gray, (2*width, 2*height), interpolation=cv.INTER_AREA)
# cv.imshow('image_1', res)
# cv.waitKey(0)
# cv.destroyAllWindows()

image_ind_2 = (image == 2)
image_ind_2_gray = (image_ind_2 * 255).astype(np.uint8)

# height, width = image_ind_2_gray.shape[:2]
# res = cv.resize(image_ind_2_gray, (2*width, 2*height),
#                 interpolation=cv.INTER_AREA)

edges_ind_2 = cv.Canny(image_ind_2_gray, 100, 200)

# height, width = edges_ind_2.shape[:2]
# res = cv.resize(edges_ind_2, (2*width, 2*height), interpolation=cv.INTER_AREA)

_, contours_ind_2, _ = cv.findContours(
    image_ind_2_gray, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

plt.figure(figsize=(7, 7))
point_cloud = []
for contour in contours_ind_2:
    points = []
    for i in contour:
        points.append(i[0].tolist())
    plt.plot(np.array(points)[:, 0], np.array(points)[:, 1], 'k.')

    point_cloud.append(np.array(points))

rect_points = []
theta_points_ind = []
for points in point_cloud:
    rect, theta = find_rectangle_param(points)
    rect_points.append(rect)
    theta_points_ind.append(theta)

#########################################

image_ind_1 = (image == 1)
image_ind_1_gray = (image_ind_1 * 255).astype(np.uint8)

height, width = image_ind_1_gray.shape[:2]
res = cv.resize(image_ind_1_gray, (2*width, 2*height),
                interpolation=cv.INTER_AREA)
# cv.imshow('image_2', res)
# cv.waitKey(0)
# cv.destroyAllWindows()

edges_ind_1 = cv.Canny(image_ind_1_gray, 100, 200)

# height, width = edges_ind_1.shape[:2]
# res = cv.resize(edges_ind_1, (2*width, 2*height), interpolation=cv.INTER_AREA)
# cv.imshow('image_3', res)
# cv.waitKey(0)
# cv.destroyAllWindows()

_, contours_ind_1, _ = cv.findContours(
    image_ind_1_gray, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

point_cloud_2 = []
for contour in contours_ind_1:
    points = []
    for i in contour:
        points.append(i[0].tolist())
    plt.plot(np.array(points)[:, 0], np.array(points)[:, 1], 'k.')
    point_cloud_2.append(np.array(points))

acc_array_contours = []
x_centroid_contour = []
y_centroid_contour = []
rho_contour = []
for points in point_cloud_2:
    acc_array, rho, theta, x_centroid, y_centroid = \
        MyHoughTransform.kernel_hough_tramsform(
            points, delta_rho, delta_theta, kernel_size, 1)

    x_centroid_contour.append(x_centroid)
    y_centroid_contour.append(y_centroid)
    acc_array_contours.append(acc_array)
    rho_contour.append(rho)

    # plt.figure()
    # plt.plot(rho, acc_array[:, theta_points_ind])
    # plt.show()

    #acc_gray = (acc_array / np.max(acc_array) * 255).astype(np.uint8)
    #scale = 2
    #height, width = acc_gray.shape[:2]
    #res = cv.resize(acc_gray, (scale * width, scale * height),
    #                interpolation=cv.INTER_AREA)
    #
    #cv.imshow('image', res)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

x_centroid_contour = np.array(x_centroid_contour)
y_centroid_contour = np.array(y_centroid_contour)

########################

contour_center = np.zeros((np.array(point_cloud_2).shape[0], 2))
i = 0
for points in point_cloud_2:
    center_x = np.sum(points[:, 0]) // points.shape[0]
    center_y = np.sum(points[:, 1]) // points.shape[0]
    if i != np.array(point_cloud_2).shape[0] - 1:
        contour_center[i, 0] = center_x
        contour_center[i, 1] = center_y
    i += 1

rect_contour = np.zeros(np.array(point_cloud).shape[0])
i = 0
for points in point_cloud:
    center_x = np.sum(points[:, 0]) // points.shape[0]
    center_y = np.sum(points[:, 1]) // points.shape[0]
    min_center = 99999
    j = 0
    for j in range(np.array(point_cloud_2).shape[0]):
        if np.sqrt((center_x - contour_center[j, 0]) ** 2 +
                   (center_y - contour_center[j, 1]) ** 2)\
                < min_center:
            rect_contour[i] = j
            min_center = np.sqrt((center_x - contour_center[j, 0]) ** 2 +
                                 (center_y - contour_center[j, 1]) ** 2)
        j += 1
    i += 1

""" require:
        acc_array_contours,
        rect_contour,
        theta_points_ind;
    out:
        magnet_lines_rho,
        magnet_lines_theta;
        """
magnet_lines_rho = np.zeros((rect_contour.shape[0], 2))
magnet_lines_theta = np.zeros((rect_contour.shape[0], 2))
for j in range(rect_contour.shape[0]):
    i = 0
    for acc_array, rho in zip(acc_array_contours, rho_contour):
        if rect_contour[j] == i:
            magnet_lines_theta[j, :] = \
                np.array([
                    theta[theta_points_ind[j]],
                    theta[theta_points_ind[j]]
                ]).reshape(1, -1)
            rho_ind = np.array((find_two_max(acc_array[:, theta_points_ind[j]])))
            # print(rho_ind)
            # plt.figure()
            # plt.plot(acc_array[:, theta_points_ind[j]])
            # plt.show()
            magnet_lines_rho[j, 0] = rho[rho_ind[0]]
            magnet_lines_rho[j, 1] = rho[rho_ind[1]]
        i += 1

magnet_lines_theta = magnet_lines_theta.reshape(-1, 1)
magnet_lines_rho = magnet_lines_rho.reshape(-1, 1)

k = -np.cos(magnet_lines_theta) / np.sin(magnet_lines_theta)
m = magnet_lines_rho / np.sin(magnet_lines_theta)

for i in range(k.shape[0]):
    ind = int(rect_contour[i // 2])

    lines = MyHoughTransform.get_lines_probabilistic(
        point_cloud_2[ind],
        magnet_lines_rho[i],
        magnet_lines_theta[i],
        kernel_size)
    for j in lines:
        plt.plot([j[0], j[1]], [j[2], j[3]], 'b')

    # x = np.array([-100, 100])
    # y = k[i] * x + m[i]
    # x += x_centroid_contour[ind]
    # y += y_centroid_contour[ind]
    # plt.plot(x, y, 'b')
    for rect in rect_points:
        plt.plot([rect[0, 0], rect[0, 1], rect[0, 2], rect[0, 3], rect[0, 0]],
            [rect[1, 0], rect[1, 1], rect[1, 2], rect[1, 3], rect[1, 0]], 'r')

plt.xlim([0, image.shape[0]])
plt.ylim([0, image.shape[0]])
plt.xlabel('x')
plt.ylabel('y')

def find_all_local_max_array(array):
    #shape_0 = array.shape[0]
    #shape_1 = array.shape[1]
    #if shape_1 % 2 == 0:
    #    new_array = np.empty((shape_0, 2 * shape_1))
    #    new_array[:, 0:shape_1 // 2] = np.flip(array[:, shape_1 // 2:], 0)
    #    new_array[:, shape_1 // 2:3 * (shape_1 // 2)] = array
    #    new_array[:, 3 * (shape_1 // 2):] = np.flip(array[:, :shape_1 // 2], 0)
    #else:
    #    new_array = np.empty((shape_0, 2 * shape_1 - 1))
    #    new_array[:, 0:shape_1 // 2] = np.flip(array[:, shape_1 // 2 + 1:], 0)
    #    new_array[:, shape_1 // 2:3 * (shape_1 // 2) + 1] = array
    #    new_array[:, 3 * (shape_1 // 2) + 1:] = np.flip(array[:, :shape_1 //
    #    2], 0)

    #array = new_array

    def dif_arr(arr, i_p, j_p):
        # 1 2 3
        # 8   4
        # 7 6 5
        dif = arr[i_p, j_p] -\
              np.array([arr[i_p - 1, j_p - 1], arr[i_p - 1, j_p    ],
                        arr[i_p - 1, j_p + 1], arr[i_p    , j_p + 1],
                        arr[i_p + 1, j_p + 1], arr[i_p + 1, j_p    ],
                        arr[i_p + 1, j_p - 1], arr[i_p    , j_p - 1],
                        ])

        return dif


    trash_hold_1 = 0.1
    max_array = np.zeros(array.shape)
    for i in range(1, array.shape[0] - 1):
        for j in range(1, array.shape[1] - 1):
            if array[i, j] > trash_hold_1 * np.max(array):
                i_pntr = i
                j_pntr = j

                dif = dif_arr(array, i_pntr, j_pntr)

                while np.min(dif) < 0:
                    dir = np.argmin(dif) + 1

                    if dir == 1:
                        i_pntr -= 1
                        j_pntr -= 1
                    elif dir == 2:
                        i_pntr -= 1
                    elif dir == 3:
                        i_pntr -= 1
                        j_pntr += 1
                    elif dir == 4:
                        j_pntr += 1
                    elif dir == 5:
                        i_pntr += 1
                        j_pntr += 1
                    elif dir == 6:
                        i_pntr += 1
                    elif dir == 7:
                        i_pntr += 1
                        j_pntr -= 1
                    elif dir == 8:
                        j_pntr -= 1

                    if i_pntr == 0 or i_pntr == array.shape[0] - 1 or\
                        j_pntr == 0 or j_pntr == array.shape[1] - 1:
                        break

                    dif = dif_arr(array, i_pntr, j_pntr)

                max_array[i_pntr, j_pntr] += 1
                # max_array[i_pntr, j_pntr] += array[i, j]

    trash_hold_2 = 0.01
    rho_ind = []
    theta_ind = []
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if max_array[i, j] > trash_hold_2 * np.max(max_array):
                rho_ind.append(i)
                theta_ind.append(j)

    #rho_ind_new = []
    #theta_ind_new = []
    #if shape_1 % 2 == 0:
    #    for rho_i, theta_i in zip(rho_ind, theta_ind):
    #        if shape_1 // 2 <= theta_i < 3 * (shape_1 // 2):
    #            rho_ind_new.append(rho_i)
    #            theta_ind_new.append(theta_i - shape_1 // 2)
    #else:
    #    for rho_i, theta_i in zip(rho_ind, theta_ind):
    #        if shape_1 // 2 <= theta_i <= 3 * (shape_1 // 2):
    #            rho_ind_new.append(rho_i)
    #            theta_ind_new.append(theta_i - shape_1 // 2)
    #
    #rho_ind = rho_ind_new
    #theta_ind = theta_ind_new

    #acc_gray = (max_array / np.max(max_array) * 255).astype(np.uint8)
    #scale = 2
    #height, width = acc_gray.shape[:2]
    #res = cv.resize(acc_gray, (scale * width, scale * height),
    #                interpolation=cv.INTER_AREA)
    #
    #cv.imshow('image', res)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    return rho_ind, theta_ind


def find_all_local_max_array_3D(array):
    def dif_arr_3D(arr, i_p, j_p, k_p):
        # 1 2 3 | 10 11 12 | 18 19 20
        # 8 9 4 | 17    13 | 25 26 21
        # 7 6 5 | 16 15 14 | 24 23 22
        dif = arr[i_p, j_p, k_p] -\
              np.array([arr[i_p - 1, j_p - 1, k_p - 1],
                        arr[i_p - 1, j_p    , k_p - 1],
                        arr[i_p - 1, j_p + 1, k_p - 1],
                        arr[i_p    , j_p + 1, k_p - 1],
                        arr[i_p + 1, j_p + 1, k_p - 1],
                        arr[i_p + 1, j_p    , k_p - 1],
                        arr[i_p + 1, j_p - 1, k_p - 1],
                        arr[i_p    , j_p - 1, k_p - 1],
                        arr[i_p    , j_p    , k_p - 1],  # 9
                        arr[i_p - 1, j_p - 1, k_p    ],
                        arr[i_p - 1, j_p    , k_p    ],
                        arr[i_p - 1, j_p + 1, k_p    ],
                        arr[i_p    , j_p + 1, k_p    ],
                        arr[i_p + 1, j_p + 1, k_p    ],
                        arr[i_p + 1, j_p    , k_p    ],
                        arr[i_p + 1, j_p - 1, k_p    ],
                        arr[i_p    , j_p - 1, k_p    ],  # 17

                        arr[i_p - 1, j_p - 1, k_p + 1],
                        arr[i_p - 1, j_p    , k_p + 1],
                        arr[i_p - 1, j_p + 1, k_p + 1],
                        arr[i_p    , j_p + 1, k_p + 1],
                        arr[i_p + 1, j_p + 1, k_p + 1],
                        arr[i_p + 1, j_p    , k_p + 1],
                        arr[i_p + 1, j_p - 1, k_p + 1],
                        arr[i_p    , j_p - 1, k_p + 1],
                        arr[i_p    , j_p    , k_p + 1]   # 26
                        ])

        return dif


    trash_hold_1 = 0.1
    max_array = np.zeros(array.shape)
    for i in range(1, array.shape[0] - 1):
        for j in range(1, array.shape[1] - 1):
            for k in range(1, array.shape[2] - 1):
                if array[i, j, k] > trash_hold_1 * np.max(array):
                    i_pntr = i
                    j_pntr = j
                    k_pntr = k

                    dif = dif_arr_3D(array, i_pntr, j_pntr, k_pntr)

                    while np.min(dif) < 0:
                        dir = np.argmin(dif) + 1

                        if dir == 1:
                            i_pntr -= 1
                            j_pntr -= 1
                            k_pntr -= 1
                        elif dir == 2:
                            i_pntr -= 1

                            k_pntr -= 1
                        elif dir == 3:
                            i_pntr -= 1
                            j_pntr += 1
                            k_pntr -= 1
                        elif dir == 4:

                            j_pntr += 1
                            k_pntr -= 1
                        elif dir == 5:
                            i_pntr += 1
                            j_pntr += 1
                            k_pntr -= 1
                        elif dir == 6:
                            i_pntr += 1

                            k_pntr -= 1
                        elif dir == 7:
                            i_pntr += 1
                            j_pntr -= 1
                            k_pntr -= 1
                        elif dir == 8:

                            j_pntr -= 1
                            k_pntr -= 1
                        elif dir == 9:


                            k_pntr -= 1
                        elif dir == 10:
                            i_pntr -= 1
                            j_pntr -= 1

                        elif dir == 11:
                            i_pntr -= 1


                        elif dir == 12:
                            i_pntr -= 1
                            j_pntr += 1

                        elif dir == 13:

                            j_pntr += 1

                        elif dir == 14:
                            i_pntr += 1
                            j_pntr += 1

                        elif dir == 15:
                            i_pntr += 1


                        elif dir == 16:
                            i_pntr += 1
                            j_pntr -= 1

                        elif dir == 17:

                            j_pntr -= 1

                        elif dir == 18:
                            i_pntr -= 1
                            j_pntr -= 1
                            k_pntr += 1
                        elif dir == 19:
                            i_pntr -= 1

                            k_pntr += 1
                        elif dir == 20:
                            i_pntr -= 1
                            j_pntr += 1
                            k_pntr += 1
                        elif dir == 21:

                            j_pntr += 1
                            k_pntr += 1
                        elif dir == 22:
                            i_pntr += 1
                            j_pntr += 1
                            k_pntr += 1
                        elif dir == 23:
                            i_pntr += 1

                            k_pntr += 1
                        elif dir == 24:
                            i_pntr += 1
                            j_pntr -= 1
                            k_pntr += 1
                        elif dir == 25:

                            j_pntr -= 1
                            k_pntr += 1
                        elif dir == 26:

                            k_pntr += 1

                        if i_pntr == 0 or i_pntr == array.shape[0] - 1 or\
                            j_pntr == 0 or j_pntr == array.shape[1] - 1 or\
                            k_pntr == 0 or k_pntr == array.shape[2] - 1:
                            break

                        dif = dif_arr_3D(array, i_pntr, j_pntr, k_pntr)

                    max_array[i_pntr, j_pntr, k_pntr] += 1

    trash_hold_2 = 0.1
    x_ind = []
    y_ind = []
    r_ind = []
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                if max_array[i, j, k] > trash_hold_2 * np.max(max_array):
                    x_ind.append(i)
                    y_ind.append(j)
                    r_ind.append(k)

    return x_ind, y_ind, r_ind


lines_2 = []
for acc_array_0, rho_contour_0, point_cloud_2_0 in \
        zip(acc_array_contours, rho_contour, point_cloud_2):
    rho_ind_0, theta_ind_0 = find_all_local_max_array(acc_array_0)
    rho_0 = rho_contour_0[rho_ind_0]
    theta_0 = theta[theta_ind_0]
    # print(rho_0)
    # print(theta_0 / np.pi * 180)

    for i in range(rho_0.shape[0]):
        lines = MyHoughTransform.get_lines_probabilistic(point_cloud_2_0,
            rho_0[i], theta_0[i], kernel_size)
        # print(lines[0])
        # print(lines[1] / np.pi * 180)
        centroid_x = np.sum(point_cloud_2_0[:, 0]) // point_cloud_2_0.shape[0]
        centroid_y = np.sum(point_cloud_2_0[:, 1]) // point_cloud_2_0.shape[0]
        if np.pi / 4 - np.pi < theta_0[i] < 3 * np.pi / 4 - np.pi or \
                np.pi / 4 < theta_0[i] < 3 * np.pi / 4 or \
                np.pi / 4 + np.pi < theta_0[i] < 3 * np.pi / 4 + np.pi:
            x = np.array([i for i in range(-1000, 1000)])
            a = np.cos(theta_0[i]) / np.sin(theta_0[i])
            b = 1
            c = - rho_0[i] / np.sin(theta_0[i])
            y = - a / b * x - c / b
            # plt.plot(x + centroid_x, y + centroid_y, 'g')
        else:
            x = np.array([i for i in range(-1000, 1000)])
            a = np.cos(theta_0[i] + np.pi / 2) / np.sin(theta_0[i] + np.pi / 2)
            b = 1
            c = - rho_0[i] / np.sin(theta_0[i] + np.pi / 2)
            y = - a / b * x - c / b
            # plt.plot(y + centroid_x, - x + centroid_y, 'g')
        for j in lines:
            lines_2.append(lines[0])

#delta_x = 5
#delta_y = 5
#delta_r = 1
#acc_circles_0, x_0, y_0, r_0, centroid_x_0, centroid_y_0 = \
#    MyHoughTransform.kernel_hough_transforn_circles(
#        point_cloud_2[2], delta_x, delta_y, delta_r, kernel_size)
#
#acc_array = acc_circles_0[:, :,
#    unravel_index(acc_circles_0.argmax(), acc_circles_0.shape)[2]]
#acc_gray = (acc_array / np.max(acc_array) * 255).astype(np.uint8)
#scale = 4
#height, width = acc_gray.shape[:2]
#res = cv.resize(acc_gray, (scale * width, scale * height),
#                interpolation=cv.INTER_AREA)
#
#cv.imshow('image', res)
#cv.waitKey(0)
#cv.destroyAllWindows()
#
#x_ind, y_ind, r_ind = find_all_local_max_array_3D(acc_circles_0)
#
#ax = plt.gca()
#
#for i in range(np.array(x_ind).shape[0]):
#    x_0_0 = x_0[x_ind[i]] + centroid_x_0
#    y_0_0 = y_0[y_ind[i]] + centroid_y_0
#    r_0_0 = r_0[r_ind[i]]
#    circle = plt.Circle((x_0_0, y_0_0), r_0_0, color='b', fill=False)
#    ax.add_artist(circle)
#    print(x_0_0)
#    print(y_0_0)
#    print(r_0_0)

points_2 = []
for points in point_cloud_2:
    for i in range(points.shape[0]):
        points_2.append([points[i, 0], points[i, 1]])
points_2 = np.array(points_2)

points_lines_num = np.zeros(len(lines_2))
dist = np.empty(len(lines_2))
for i in range(points_2.shape[0]):
    j = 0
    for l in lines_2:
        if np.abs(l[3] - l[2]) < np.abs(l[1] - l[0]):
            a = - np.arctan((l[3] - l[2]) / (l[1] - l[0]))
            b = 1
            c = l[0] * np.tan(- a) + l[2]
            x_0 = points_2[i, 0]
            y_0 = points_2[i, 1]
            dist[j] = np.abs(a * x_0 + b * y_0 + c) / np.sqrt(a ** 2 + b ** 2)
        else:
            a = - np.arctan((l[1] - l[0]) / (- l[3] + l[2]))
            b = 1
            c = - l[2] * np.tan(- a) + l[0]
            x_0 = - points_2[i, 1]
            y_0 = points_2[i, 0]
            dist[j] = np.abs(a * x_0 + b * y_0 + c) / np.sqrt(a ** 2 + b ** 2)

        x_0 = points_2[i, 0]
        y_0 = points_2[i, 1]
        p = np.tan(- a)
        m = 1
        t = (m * x_0 + p * y_0 - m * l[0] - p * l[2]) / (m ** 2 + p ** 2)
        x_proj = m * t + l[0]
        y_proj = p * t + l[2]

        if l[0] < x_proj < l[1] or l[0] > x_proj > l[1]:
            dist[j] = np.min(np.array([
                np.sqrt((l[0]**2 - x_0**2) + (l[2]**2 - y_0**2)),
                np.sqrt((l[1]**2 - x_0**2) + (l[3]**2 - y_0**2))
            ]))

        j += 1

    points_lines_num[np.argmin(dist)] += 1

print(points_lines_num.tolist())
for l, num in zip(lines_2, points_lines_num.tolist()):
    if num > 0:
        plt.plot([l[0], l[1]], [l[2], l[3]], 'y')

print("--- %s seconds ---" % (time.time() - start_time))

plt.show()
