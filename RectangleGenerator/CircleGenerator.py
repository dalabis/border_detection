import numpy as np
import random


def circle_gen(radius_range, center_x_range, center_y_range, point_num_range,
    random_radius_range):

    def circle_split(radius, x_center, y_center, point_num):
        theta = np.array(range(30)) / 30 * 2 * np.pi
        x_split = np.zeros(point_num)
        y_split = np.zeros(point_num)

        for i in range(30):
            x_split = center_x + radius * np.cos(theta)
            y_split = center_y + radius * np.sin(theta)

        return x_split, y_split

    def circle_random(x_split, y_split, random_radius):
        x_rand = np.zeros(x_split.shape[0])
        y_rand = np.zeros(y_split.shape[0])

        for i in range(x_split.shape[0]):
            x_rand[i] = random.uniform(x_split[i] - random_radius, x_split[i] + random_radius)
            y_rand[i] = random.uniform(y_split[i] - random_radius, y_split[i] + random_radius)

        return x_rand, y_rand

    while True:
        radius = random.uniform(radius_range[0], radius_range[1])
        center_x = random.uniform(center_x_range[0], center_x_range[1])
        center_y = random.uniform(center_y_range[0], center_y_range[1])
        point_num = int(random.uniform(point_num_range[0], point_num_range[1]))
        random_radius = random.uniform(random_radius_range[0], random_radius_range[1])

        param = [radius, center_x, center_y]

        x_split, y_split = circle_split(radius, center_x, center_y, point_num)
        x_rand, y_rand = circle_random(x_split, y_split, random_radius)

        yield x_rand, y_rand, param
