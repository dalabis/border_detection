import MyHoughTransform

import cv2
import numpy as np
from numpy import unravel_index
import pandas as pd
from matplotlib import pyplot as plt


def find_two_max(rho_line):
    ind_line_1 = 0
    ind_line_2 = 0
    rho_line_1 = 0
    for i in range(1, rho_line.shape[0] - 1):
        if rho_line[i] > rho_line[i - 1] and rho_line[i] > rho_line[i + 1] and \
                rho_line[i] > rho_line_1:
            temp = ind_line_1
            ind_line_1 = i
            ind_line_2 = temp
            rho_line_1 = rho_line[i]

    return [ind_line_1, ind_line_2]


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


data = pd.read_table(
    'input/three_examples/1.txt', delim_whitespace=True, names=('x', 'y'))

plt.plot(data['x'], data['y'], '.')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

points = np.array(data)
delta_rho = 0.1
delta_theta = np.pi / 180
kernel_size = 0.5
acc_array, rho, theta, x_centroid, y_centroid = \
    MyHoughTransform.kernel_hough_tramsform(
        points, delta_rho, delta_theta, kernel_size)

acc_gray = (acc_array / np.max(acc_array) * 255).astype(np.uint8)
scale = 4
height, width = acc_gray.shape[:2]
res = cv2.resize(acc_gray, (scale * width, scale * height),
                 interpolation=cv2.INTER_AREA)

cv2.imshow('image', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

theta_long_line_ind = unravel_index(acc_array.argmax(), acc_array.shape)[1]
if theta_long_line_ind + acc_array.shape[1] // 2 > acc_array.shape[1]:
    theta_short_line_ind = theta_long_line_ind - acc_array.shape[1] // 2
else:
    theta_short_line_ind = theta_long_line_ind + acc_array.shape[1] // 2

pho_long_line = acc_array[:, theta_long_line_ind]
pho_short_line = acc_array[:, theta_short_line_ind]

plt.figure()
plt.plot(rho, pho_long_line)
plt.plot(rho, pho_short_line)
plt.show()

long_lines = find_two_max(pho_long_line)
short_lines = find_two_max(pho_short_line)

rho_lines = np.array([rho[long_lines[0]], rho[short_lines[0]],
                      rho[long_lines[1]], rho[short_lines[1]]])
theta_lines = np.array([theta[theta_long_line_ind], theta[theta_short_line_ind],
                        theta[theta_long_line_ind], theta[theta_short_line_ind]])

result_points = find_intersections(rho_lines, theta_lines)

result_points[0, :] += x_centroid
result_points[1, :] += y_centroid

plt.figure(figsize=(5, 5))
plt.plot(data['x'], data['y'], 'k.')
plt.plot(
    [result_points[0, 0], result_points[0, 1], result_points[0, 2],
     result_points[0, 3], result_points[0, 0]],
    [result_points[1, 0], result_points[1, 1], result_points[1, 2],
     result_points[1, 3], result_points[1, 0]]
)
x_max = np.max(data['x'])
x_min = np.min(data['x'])
y_max = np.max(data['y'])
y_min = np.min(data['y'])
delt = np.array([x_max - x_min, y_max - y_min])
plt.xlim([x_min, x_min + np.max(delt)])
plt.ylim([y_min, y_min + np.max(delt)])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
