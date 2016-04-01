# coding=utf8

import numpy as np

elastic_force = lambda dx, k: -k * dx
centrifugal_force = lambda x, x0, k: k * (x - x0)


denomin_gravit_force = lambda x, x0: ((x[0] - x0[0])**2 + (x[1] - x0[1])**2)**(3.0/2.0)
gravitational_force = lambda x, x0, k: -k * (x - x0) / denomin_gravit_force(x, x0)

x = np.array([5, 3])
x0 = np.array([3, 2])

print elastic_force(x0, 0.5)
print centrifugal_force(x, x0, 0.5)

print denomin_gravit_force(x, x0)
print gravitational_force(x, x0, 0.5)