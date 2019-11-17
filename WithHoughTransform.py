import cv2
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import Normalizer

from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_table(
    'input/three_examples/1.txt', delim_whitespace=True, names=('x', 'y'))

plt.plot(data['x'], data['y'], '.')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Построение пиксельного изображения
hist = np.histogram2d(data['x'], data['y'], bins=200)

# Размытие точек с помощью гауссового фильтра
hist_gauss = cv2.GaussianBlur(hist[0], (5, 5), cv2.BORDER_DEFAULT)
# Нормализация изображения
im_gray = (hist_gauss / np.max(hist_gauss) * 255).astype(np.uint8)
# Цветное изображение
im_rgb = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2RGB)
# Бинаризация
im_bw = cv2.threshold(im_gray, 1, 255, cv2.THRESH_BINARY)[1]

# Применение алгоритма Хафа для поиска прямых
lines = cv2.HoughLines(im_bw, 1, np.pi/180, 15)

# Отображение найденных прямых
rho = []
theta = []
for line in lines:
    rho.append(line[0][0])
    theta.append(line[0][1])
    a = np.cos(theta[-1])
    b = np.sin(theta[-1])
    x0 = a * rho[-1]
    y0 = b * rho[-1]
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)

    cv2.line(im_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('image', im_rgb)
cv2.imwrite('hough.png', im_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.plot(rho, theta, '.')
plt.xlabel('r, pix')
plt.ylabel('theta, rad')
plt.show()

# Разбиение кривых на четыре класса
X = np.column_stack((rho, [i * 100 for i in theta]))
AC = AgglomerativeClustering(n_clusters=4).fit(X)

sns.scatterplot(x=rho, y=theta, hue=AC.labels_)
plt.xlabel('r, pix')
plt.ylabel('theta, rad')
plt.show()

# Из каждого класса берется первая прямая с начала массива lines, потому что
# HoughLines возвращает прямые в порядке убывания их значения из
# аккумуляторного массива
result_lines = []
for i in range(4):
    for line, label in zip(lines, AC.labels_):
        if label == i:
            result_lines.append(line)
            break

# Зеленые линии соответвуют выбраным прямым
for line in result_lines:
    rho.append(line[0][0])
    theta.append(line[0][1])
    a = np.cos(theta[-1])
    b = np.sin(theta[-1])
    x0 = a * rho[-1]
    y0 = b * rho[-1]
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)

    cv2.line(im_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('image', im_rgb)
cv2.imwrite('result.png', im_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Переход к представлению y = k * x + m
k = np.zeros(4)
m = np.zeros(4)
for i in range(4):
    k[i] = - np.cos(result_lines[i][0][1]) / np.sin(result_lines[i][0][1])
    m[i] = result_lines[i][0][0] / np.sin(result_lines[i][0][1])

# Далее идет сортировка прямых с целью найти четыре искомые точки
for i in range(4):
    for j in range(3):
        if k[j] < k[j + 1]:
            temp = k[j]
            k[j] = k[j + 1]
            k[j + 1] = temp
            temp = m[j]
            m[j] = m[j + 1]
            m[j + 1] = temp

if m[0] < m[1]:
    temp = k[0]
    k[0] = k[1]
    k[1] = temp
    temp = m[0]
    m[0] = m[1]
    m[1] = temp

if m[2] < m[3]:
    temp = k[2]
    k[2] = k[3]
    k[3] = temp
    temp = m[2]
    m[2] = m[3]
    m[3] = temp

temp = k[1]
k[1] = k[2]
k[2] = temp
temp = m[1]
m[1] = m[2]
m[2] = temp

print(k)
print(m)

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

# Переход из координат на пиксельном изображении к исходным координатам
result_final = np.zeros((2, 4))
result_final[0, :] = result_points[1, :] /\
    200 * (hist[1][-1] - hist[1][0]) + hist[1][0]
result_final[1, :] = result_points[0, :] /\
    200 * (hist[2][-1] - hist[2][0]) + hist[2][0]

# Искомый четырехугольник, противолежащие грани не обязательно параллельны
plt.plot(data['x'], data['y'], 'k.')
plt.plot(
    [result_final[0, 0], result_final[0, 1], result_final[0, 2],
    result_final[0, 3], result_final[0, 0]],
    [result_final[1, 0], result_final[1, 1], result_final[1, 2],
    result_final[1, 3], result_final[1, 0]]
)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
