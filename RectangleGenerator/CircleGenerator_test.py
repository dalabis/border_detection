import CircleGenerator

from matplotlib import pyplot as plt

circ_gen = CircleGenerator.circle_gen(
    radius_range=[5, 10],
    center_x_range=[-5, 5],
    center_y_range=[-5, 5],
    point_num_range=[30, 40],
    random_radius_range=[0.2, 0.4]
)

x_rand, y_rand, _ = next(circ_gen)

plt.figure(figsize=(5, 5))
plt.plot(x_rand, y_rand, '.')
plt.xlim([-15, 15])
plt.ylim([-15, 15])
plt.show()
