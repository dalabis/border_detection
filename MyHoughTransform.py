import numpy as np
import copy


def kernel_hough_tramsform(points, delta_rho, delta_theta, kernel_size):
    centroid_x = np.sum(points[:, 0]) // points.shape[0]
    centroid_y = np.sum(points[:, 1]) // points.shape[0]

    points_0 = copy.copy(points)
    points_0[:, 0] = points[:, 0] - centroid_x
    points_0[:, 1] = points[:, 1] - centroid_y

    points_rho = np.sqrt(points_0[:, 0] ** 2 + points_0[:, 1] ** 2)
    min_rho = -np.max(points_rho) - delta_rho
    max_rho = np.max(points_rho) + delta_rho
    num_rho = int((max_rho - min_rho) / delta_rho)
    rho = np.array([min_rho + i * delta_rho for i in range(num_rho)])

    # points_theta = np.arctan(points[:, 0] / points[:, 1]) + np.pi / 2
    min_theta = delta_theta
    max_theta = np.pi
    num_theta = int((max_theta - min_theta) / delta_theta)
    theta = np.array([min_theta + i * delta_theta for i in range(num_theta)])

    acc_array = np.zeros((num_rho, num_theta))

    for i in range(num_rho):
        for j in range(num_theta):
            a = np.cos(theta[j]) / np.sin(theta[j])
            b = 1
            c = -rho[i] / np.sin(theta[j])
            x_0 = points_0[:, 0]
            y_0 = points_0[:, 1]
            dist = np.abs(a * x_0 + b * y_0 + c) / np.sqrt(a ** 2 + b ** 2)
            acc_array[i, j] = np.sum(1 / (2 * np.pi * kernel_size ** 2) * \
                np.exp(-(dist ** 2) / (2 * kernel_size ** 2)))

    # where_are_NaNs = np.isnan(acc_array)
    # acc_array[where_are_NaNs] = 0

    # r_0 = np.sqrt(centroid_x ** 2 + centroid_y ** 2)
    # theta_0 = np.arctan(centroid_y / centroid_x)
    # rho += r_0 * np.cos(theta_0 - theta - np.pi / 2)

    return acc_array, rho, theta, centroid_x, centroid_y


def get_lines_probabilistic(points, rho, theta, kernel_size):
    min_gap = kernel_size * 5
    trash_hold = 0.5

    centroid_x = np.sum(points[:, 0]) // points.shape[0]
    centroid_y = np.sum(points[:, 1]) // points.shape[0]

    points_0 = copy.copy(points)
    points_0[:, 0] = points[:, 0] - centroid_x
    points_0[:, 1] = points[:, 1] - centroid_y

    r = np.array([i for i in range(-100, 100)])
    x = r * np.cos(theta + np.pi/2) + rho / np.cos(theta)
    y = r * np.sin(theta + np.pi/2)

    acc = np.zeros(r.shape[0])

    for i in range(r.shape[0]):
        x_0 = points_0[:, 0]
        y_0 = points_0[:, 1]

        dist = np.sqrt((x[i] - x_0) ** 2 + (y[i] - y_0) ** 2)
        acc[i] = np.sum(1 / (2 * np.pi * kernel_size ** 2) * \
            np.exp(-(dist ** 2) / (2 * kernel_size ** 2)))

    lines = []
    line_success = 0
    for i in range(acc.shape[0]):
        if acc[i] > trash_hold * np.max(acc) and line_success == 0:
            line_success = 1
            x_1 = x[i] + centroid_x
            y_1 = y[i] + centroid_y
        elif acc[i] < trash_hold * np.max(acc) and line_success == 1:
            line_success = 0
            x_2 = x[i] + centroid_x
            y_2 = y[i] + centroid_y
            lines.append([x_1, x_2, y_1, y_2])
        elif i == acc.shape[0] - 1 and line_success == 1:
            x_2 = x[i] + centroid_x
            y_2 = y[i] + centroid_y
            lines.append([x_1, x_2, y_1, y_2])

    lines_out = []
    for i in lines:
        if np.sqrt((i[0] - i[1]) ** 2 + (i[2] - i[3]) ** 2) > min_gap:
            lines_out.append(i)

    return lines_out


def kernel_hough_transforn_circles(
        points, delta_x, delta_y, delta_r, kernel_size):
    centroid_x = np.sum(points[:, 0]) // points.shape[0]
    centroid_y = np.sum(points[:, 1]) // points.shape[0]

    points_0 = copy.copy(points)
    points_0[:, 0] = points[:, 0] - centroid_x
    points_0[:, 1] = points[:, 1] - centroid_y

    # points_rho = np.sqrt(points_0[:, 0] ** 2 + points_0[:, 1] ** 2)
    # min_rho = -np.max(points_rho) - delta_rho
    # max_rho = np.max(points_rho) + delta_rho
    # num_rho = int((max_rho - min_rho) / delta_rho)
    # rho = np.array([min_rho + i * delta_rho for i in range(num_rho)])

    min_x = -100
    max_x = 100
    num_x = (max_x - min_x) // delta_x
    x = np.array([min_x + i * delta_x for i in range(num_x)])

    min_y = -100
    max_y = 100
    num_y = (max_y - min_y) // delta_y
    y = np.array([min_y + i * delta_y for i in range(num_y)])

    min_r = 0
    max_r = 20
    num_r = (max_r - min_r) // delta_r
    r = np.array([min_r + i * delta_r for i in range(num_r)])

    acc_array = np.zeros((num_x, num_y, num_r))

    shape_0 = points_0.shape[0]
    shape_1 = x.shape[0]
    shape_2 = y.shape[0]
    shape_3 = r.shape[0]
    shape = (shape_0, shape_1, shape_2, shape_3)

    x_arr = np.zeros(shape)
    y_arr = np.zeros(shape)
    r_arr = np.zeros(shape)
    x_0_arr = np.zeros(shape)
    y_0_arr = np.zeros(shape)
    for i in range(shape_0):
        for j in range(shape_1):
            for k in range(shape_2):
                for l in range(shape_3):
                    x_arr[i, j, k, l] = x[j]
                    y_arr[i, j, k, l] = y[k]
                    r_arr[i, j, k, l] = r[l]
                    x_0_arr[i, j, k, l] = points_0[i, 0]
                    y_0_arr[i, j, k, l] = points_0[i, 1]

    dist = np.abs(np.sqrt((x_0_arr - x_arr) ** 2 + (y_0_arr - y_arr) ** 2) - r_arr)
    acc_array = np.sum(1 / (2 * np.pi * kernel_size ** 2) *
        np.exp(-(dist ** 2) / (2 * kernel_size ** 2)), axis=0)

    # where_are_NaNs = np.isnan(acc_array)
    # acc_array[where_are_NaNs] = 0

    # r_0 = np.sqrt(centroid_x ** 2 + centroid_y ** 2)
    # theta_0 = np.arctan(centroid_y / centroid_x)
    # rho += r_0 * np.cos(theta_0 - theta - np.pi / 2)

    return acc_array, x, y, r, centroid_x, centroid_y

def get_lines_probabilistic(points, rho, theta, kernel_size):
    min_gap = kernel_size * 5
    trash_hold = 0.5

    centroid_x = np.sum(points[:, 0]) // points.shape[0]
    centroid_y = np.sum(points[:, 1]) // points.shape[0]

    points_0 = copy.copy(points)
    points_0[:, 0] = points[:, 0] - centroid_x
    points_0[:, 1] = points[:, 1] - centroid_y

    r = np.array([i for i in range(-100, 100)])
    x = r * np.cos(theta + np.pi/2) + rho / np.cos(theta)
    y = r * np.sin(theta + np.pi/2)

    acc = np.zeros(r.shape[0])

    for i in range(r.shape[0]):
        x_0 = points_0[:, 0]
        y_0 = points_0[:, 1]

        dist = np.sqrt((x[i] - x_0) ** 2 + (y[i] - y_0) ** 2)
        acc[i] = np.sum(1 / (2 * np.pi * kernel_size ** 2) * \
            np.exp(-(dist ** 2) / (2 * kernel_size ** 2)))

    lines = []
    line_success = 0
    for i in range(acc.shape[0]):
        if acc[i] > trash_hold * np.max(acc) and line_success == 0:
            line_success = 1
            x_1 = x[i] + centroid_x
            y_1 = y[i] + centroid_y
        elif acc[i] < trash_hold * np.max(acc) and line_success == 1:
            line_success = 0
            x_2 = x[i] + centroid_x
            y_2 = y[i] + centroid_y
            lines.append([x_1, x_2, y_1, y_2])
        elif i == acc.shape[0] - 1 and line_success == 1:
            x_2 = x[i] + centroid_x
            y_2 = y[i] + centroid_y
            lines.append([x_1, x_2, y_1, y_2])

    lines_out = []
    for i in lines:
        if np.sqrt((i[0] - i[1]) ** 2 + (i[2] - i[3]) ** 2) > min_gap:
            lines_out.append(i)

    return lines_out
