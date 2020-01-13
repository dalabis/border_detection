import RectangleGenerator

import pandas as pd

point_num = 30
rectangle_num = 1000

rect_gen = RectangleGenerator.rectangle_gen(
    height_range=[5, 15],
    width_range=[10, 20],
    center_x_range=[-5, 5],
    center_y_range=[-5, 5],
    rotation_range=[0, 360],
    point_num_range=[point_num, point_num],
    random_radius_range=[0.2, 0.4]
)

df = pd.DataFrame(
    columns=
        ['x_in_' + str(i) for i in range(point_num)] +
        ['y_in_' + str(i) for i in range(point_num)] +
        ['x_out_' + str(i) for i in range(4)] +
        ['y_out_' + str(i) for i in range(4)] +
        ['height', 'width', 'center_x', 'center_y', 'rotation']
)

for i, rect in enumerate(rect_gen):
    if i >= rectangle_num:
        break

    df.loc[i] = rect[0].tolist() + rect[1].tolist() + rect[2].tolist() + rect[3].tolist() + rect[4]

df.to_csv(r'D:\Bosch\input\rectangle_sets\rectangle_set_1000_30_points.csv')
