import RectangleGenerator
import CircleGenerator

import random

from matplotlib import pyplot as plt

import pandas as pd

point_num = 30
samples_num = 1000

rect_gen = RectangleGenerator.rectangle_gen(
    height_range=[5, 15],
    width_range=[10, 20],
    center_x_range=[-5, 5],
    center_y_range=[-5, 5],
    rotation_range=[0, 360],
    point_num_range=[point_num, point_num],
    random_radius_range=[0.2, 0.4]
)

circ_gen = CircleGenerator.circle_gen(
    radius_range=[5, 10],
    center_x_range=[-5, 5],
    center_y_range=[-5, 5],
    point_num_range=[point_num, point_num],
    random_radius_range=[0.2, 0.4]
)

df = pd.DataFrame(
    columns=
        ['x' + str(i) for i in range(point_num)] +
        ['y' + str(i) for i in range(point_num)] +
        ['target' + str(i) for i in range(point_num)]
)

for i in range(samples_num):
    if random.random() > 0.5:
        shape = next(circ_gen)
        target = [1 for i in range(point_num)]
    else:
        shape = next(rect_gen)
        target = [0 for i in range(point_num)]

    df.loc[i] = shape[0].tolist() + shape[1].tolist() + target

df.to_csv(r'D:\Bosch\input\rectangle_sets\rectangle_or_circle_set_1000_30_points.csv')
