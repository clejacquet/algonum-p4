# coding=utf8

import numpy as np

k = 0.5
x0 = np.array([3, 2])

denomin_gravit_force = lambda x: ((x[0] - x0[0])**2 + (x[1] - x0[1])**2)**(3.0/2.0)


elastic_force = lambda dx: -k * dx
centrifugal_force = lambda x: k * (x - x0)
gravitational_force = lambda x: -k * (x - x0) / denomin_gravit_force(x)


elastic_jacob = np.array([[- k, 0],
                          [0, -k]])

centrifugal_jacob = np.array([[k, 0],
                              [0, k]])

# TO DO
gravitational_jacob = np.array([[0, 0],
                                [0, 0]])

x = np.array([5, 3])

print elastic_force(x)
print centrifugal_force(x)
print gravitational_force(x)
