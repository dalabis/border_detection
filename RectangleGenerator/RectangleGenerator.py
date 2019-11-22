import numpy as np
import random


def rectangle_gen(
    height_range,
    width_range,
    center_x_range,
    center_y_range,
    rotation_range,
    point_num_range,
    random_radius_range
):

    def rectangle_transform(height, width, center_x, center_y, rotation):
        # This function returns the coordinates of the corners of the rectangle

        # Coordinates initialization
        x = np.zeros(4)
        y = np.zeros(4)

        # Coordinates without rotation
        # Anticlockwise order starting from the lower left corner
        x[0] = center_x - width / 2
        x[1] = center_x + width / 2
        x[2] = center_x + width / 2
        x[3] = center_x - width / 2
        y[0] = center_y - height / 2
        y[1] = center_y - height / 2
        y[2] = center_y + height / 2
        y[3] = center_y + height / 2

        x_rot = np.zeros(4)
        y_rot = np.zeros(4)

        # Coordinates with rotation
        # Clockwise rotation
        theta = rotation / 180 * np.pi
        for i in range(4):
            x_rot[i] = (x[i] - center_x) * np.cos(theta) - \
                       (y[i] - center_y) * np.sin(theta) + center_x
            y_rot[i] = (x[i] - center_x) * np.sin(theta) + \
                       (y[i] - center_y) * np.cos(theta) + center_y

        return x_rot, y_rot

    def rectangle_split(x, y, point_num):
        side_0 = np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)
        side_1 = np.sqrt((x[1] - x[2]) ** 2 + (y[1] - y[2]) ** 2)
        side_2 = np.sqrt((x[2] - x[3]) ** 2 + (y[2] - y[3]) ** 2)
        side_3 = np.sqrt((x[3] - x[0]) ** 2 + (y[3] - y[0]) ** 2)
        perimeter = side_0 + side_1 + side_2 + side_3
        step = perimeter / point_num
        counter = 0
        x_split = np.zeros(point_num)
        y_split = np.zeros(point_num)
        residial = np.zeros(3)
        success = np.zeros(3)

        x_split[0] = x[0]
        y_split[0] = y[0]
        for i in range(1, point_num):
            if counter + step < side_0 and success[0] == 0:
                x_split[i] = (x[1] - x[0]) * (counter + step) / side_0 + x[0]
                y_split[i] = (y[1] - y[0]) * (counter + step) / side_0 + y[0]
                counter += step
                continue
            elif success[0] == 0:
                residial[0] = side_0 - counter
                counter = 0
                success[0] = 1
            if counter + step - residial[0] < side_1 and success[1] == 0:
                x_split[i] = (x[2] - x[1]) * (counter + step - residial[0]) / side_1 + x[1]
                y_split[i] = (y[2] - y[1]) * (counter + step - residial[0]) / side_1 + y[1]
                counter += step
                continue
            elif success[1] == 0:
                residial[1] = side_1 - counter + residial[0]
                counter = 0
                success[1] = 1
            if counter + step - residial[1] < side_2 and success[2] == 0:
                x_split[i] = (x[3] - x[2]) * (counter + step - residial[1]) / side_2 + x[2]
                y_split[i] = (y[3] - y[2]) * (counter + step - residial[1]) / side_2 + y[2]
                counter += step
                continue
            elif success[2] == 0:
                residial[2] = side_2 - counter + residial[1]
                counter = 0
                success[2] = 1
            if counter + step - residial[2] < side_3:
                x_split[i] = (x[0] - x[3]) * (counter + step - residial[2]) / side_3 + x[3]
                y_split[i] = (y[0] - y[3]) * (counter + step - residial[2]) / side_3 + y[3]
                counter += step
                continue

        return x_split, y_split

    def rectangle_random(x_split, y_split, random_radius):
        x_rand = np.zeros(x_split.shape[0])
        y_rand = np.zeros(y_split.shape[0])

        for i in range(x_split.shape[0]):
            x_rand[i] = random.uniform(x_split[i] - random_radius, x_split[i] + random_radius)
            y_rand[i] = random.uniform(y_split[i] - random_radius, y_split[i] + random_radius)

        return x_rand, y_rand

    while True:
        height = random.uniform(height_range[0], height_range[1])
        width = random.uniform(width_range[0], width_range[1])
        center_x = random.uniform(center_x_range[0], center_x_range[1])
        center_y = random.uniform(center_y_range[0], center_y_range[1])
        rotation = random.uniform(rotation_range[0], rotation_range[1])
        point_num = int(random.uniform(point_num_range[0], point_num_range[1]))
        random_radius = random.uniform(random_radius_range[0], random_radius_range[1])

        x, y = rectangle_transform(height, width, center_x, center_y, rotation)
        x_split, y_split = rectangle_split(x, y, point_num)
        x_rand, y_rand = rectangle_random(x_split, y_split, random_radius)

        yield x_rand, y_rand, x, y
