# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# # 3d Surface
# fig = plt.figure()
#
# ax = fig.add_subplot(111, projection='3d')
#
# # make data
# u = np.linspace(0, 2 * np.pi, 100) # 0 ~ 2pi sample 100 points
# v = np.linspace(0, np.pi, 100) # 0 ~ pi, sample 100 points
# x = 10 * np.outer(np.cos(u), np.sin(v))
# y = 10 * np.outer(np.sin(u), np.sin(v))
# z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
#
# ax.plot_surface(x, y, z, color='b')
#
# plt.show()
#
#
#
# import numpy as np
#
# pose2d = np.arange(3*4*3).reshape(3,4,3)
# print(pose2d)
#
# print('test \n\n ')
# print(pose2d[:, 0])
# print('test \n\n ')
# print(pose2d[:, 2])
# print('test \n\n ')
# temp = np.stack([pose2d[:, 0], pose2d[:, 2]], axis=-1)
# print('test \n\n ')
# print(temp)

from mpl_toolkits.mplot3d import Axes3D



# Thomas Liao created Sep 2018, demo for Fibonacci Lattice
import matplotlib.pyplot as plt
import numpy as np

required_num_of_points = 1000
N = required_num_of_points * 2
phi = 0.5*(np.sqrt(5) - 1)
n = np.arange(1, N+1)
# Upper Hemisphere
n = n[N//2:]

# Fibonacci Grid
zn = (2*n - 1)/N - 1
xn = np.sqrt(-zn**2 + 1) * np.cos(2*np.pi*n*phi)
yn = np.sqrt(-zn**2 + 1) * np.sin(2*np.pi*n*phi)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xn, yn, zn, color='b')

plt.show()




