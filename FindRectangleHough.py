import cv2
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
import seaborn as sns


def find_rectangle(data):
    # Построение пиксельного изображения
    hist = np.histogram2d(data[:, 0], data[:, 1], bins=200)

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

    # Разбиение кривых на четыре класса
    X = np.column_stack((rho, [i * 100 for i in theta]))
    AC = AgglomerativeClustering(n_clusters=4).fit(X)

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

    return result_final


def find_polygon(data):
    # Построение пиксельного изображения
    scale = 2
    data = data * scale
    bins = 1 + np.max(np.array([np.max(data[:, 0]) - np.min(data[:, 0]),
                                np.max(data[:, 1]) - np.min(data[:, 1])]))
    #hist = np.histogram2d(data[:, 0], data[:, 1], bins=bins)
    #hist_square = np.zeros((bins, bins))
    #hist_square[:hist[0].shape[0], :hist[0].shape[1]] = hist[0]
    #if hist[1] =
    image = np.zeros((bins, bins))
    x_min = np.min(data[:, 0])
    y_min = np.min(data[:, 1])
    x_range = np.array([x_min + i for i in range(bins)])
    y_range = np.array([y_min + i for i in range(bins)])
    for i in range(data.shape[0]):
        image[data[i, 0] - x_min, data[i, 1] - y_min] = 1

    # Размытие точек с помощью гауссового фильтра
    hist_gauss = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
    #hist_gauss = image
    # Нормализация изображения
    im_gray = (hist_gauss / np.max(hist_gauss) * 255).astype(np.uint8)
    # Цветное изображение
    im_rgb = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2RGB)
    # Бинаризация
    im_bw = cv2.threshold(im_gray, 1, 255, cv2.THRESH_BINARY)[1]

    # Применение алгоритма Хафа для поиска прямых
    lines = cv2.HoughLinesP(im_bw, 1, np.pi/180*1,
                            threshold=40, minLineLength=15, maxLineGap=5)

    # Отображение найденных прямых
    rho = []
    theta = []
    for line in lines:
        l = line[0]

        theta_temp = np.arctan((l[3] - l[2]) / (l[1] - l[0])) - np.pi / 2
        theta.append(theta_temp)
        rho_temp = l[0] * np.cos(theta_temp) + l[2] * np.sin(theta_temp)
        rho.append(rho_temp)

        cv2.line(im_rgb, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3,
                 cv2.LINE_AA)

    #    cv2.line(im_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)

    #cv2.imshow('image', im_rgb)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    X = np.column_stack((rho, theta))
    X = StandardScaler().fit_transform(X)
    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=2).fit(X)

    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    #plt.figure()
    #sns.scatterplot(x=rho, y=theta, hue=labels)
    #plt.xlabel('r, pix')
    #plt.ylabel('theta, rad')
    #plt.show()

    # Из каждого класса берется первая прямая с начала массива lines, потому что
    # HoughLines возвращает прямые в порядке убывания их значения из
    # аккумуляторного массива
    result_lines = []
    for i in range(n_clusters_):
        for line, label in zip(lines, labels):
            if label == i:
                result_lines.append(line)
                break

    print(n_clusters_)

    result_points = []
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

        l = line[0]
        result_points.append(np.array(l))

        cv2.line(im_rgb, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 2)

    cv2.imshow('image', im_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    result_points = np.array(result_points)
    result_final = np.zeros((n_clusters_, 4))
    result_final[:, 0] = \
        result_points[:, 1] / bins * (x_range[-1] - x_range[0]) + x_range[0]
    result_final[:, 1] = \
        result_points[:, 0] / bins * (y_range[-1] - y_range[0]) + y_range[0]
    result_final[:, 2] = \
        result_points[:, 3] / bins * (x_range[-1] - x_range[0]) + x_range[0]
    result_final[:, 3] = \
        result_points[:, 2] / bins * (y_range[-1] - y_range[0]) + y_range[0]

    return result_final / scale
