import RectangleGenerator

from matplotlib import pyplot as plt

rect_gen = RectangleGenerator.rectangle_gen(
    height_range=[5, 15],
    width_range=[10, 20],
    center_x_range=[-5, 5],
    center_y_range=[-5, 5],
    rotation_range=[0, 360],
    point_num_range=[30, 40],
    random_radius_range=[0.2, 0.4]
)

x_rand, y_rand, x, y = next(rect_gen)

plt.figure(figsize=(5, 5))
plt.plot(x_rand, y_rand, '.')
plt.plot(x, y, '.')
plt.xlim([-15, 15])
plt.ylim([-15, 15])
plt.show()
