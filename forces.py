# coding=utf8

import numpy as np
import NRMethod as nr


denomin_gravit_force = lambda x, x0: ((x[0] - x0[0])**2 + (x[1] - x0[1])**2)**(3.0/2.0)


elastic_force = lambda x, x0, k: -k * x
centrifugal_force = lambda x, x0, k: k * (x - x0)
gravitational_force = lambda x, x0, k: -k * (x - x0) / denomin_gravit_force(x, x0)


elastic_jacob = lambda x, x0, k: np.array([[-k,0],
                          [0, -k]])

centrifugal_jacob =lambda x, x0, k: np.array([[k,0],
                              [0, k]])



gravitational_jacob = lambda x,x0,k: np.array([[-k*(denomin_gravit_force(x,x0)-((x[0]-x0[0])**2)*3*(((x[0] - x0[0])**2) + ((x[1] - x0[1])**2))**(1.0/2.0))/(denomin_gravit_force(x,x0)**2) , -k*(-(x[0]-x0[0])*3*(x[1]-x0[1])*(((x[0] - x0[0])**2) + ((x[1] - x0[1])**2))**(1.0/2.0))/(denomin_gravit_force(x,x0)**2)],
                                               [-k*(-(x[1]-x0[1])*3*(x[0]-x0[0])*(((x[0] - x0[0])**2) + ((x[1] - x0[1])**2))**(1.0/2.0))/(denomin_gravit_force(x,x0)**2) , -k*(denomin_gravit_force(x,x0)-((x[1]-x0[1])**2)*3*(((x[0] - x0[0])**2) + ((x[1] - x0[1])**2))**(1.0/2.0))/(denomin_gravit_force(x,x0)**2)]])



def f_gen(force_list):
    def f(vec):
        total = np.array([0., 0.])
        for i in range(0, len(force_list)):
            force = force_list[i][0]
            origin = force_list[i][1]
            coef = force_list[i][2]

            res = force(vec, origin, coef)
            total += res
        return total
    return f


def df_gen(jacob_list):
    def df(vec):
        total = np.array([[0., 0.],
                          [0., 0.]])
        for i in range(0, len(jacob_list)):
            jacob = jacob_list[i][0]
            origin = jacob_list[i][1]
            coef = jacob_list[i][2]

            res = jacob(vec, origin, coef)
            total += res
        return total
    return df

x = np.array([1.5, 0])
x0 = np.array([0, 0])
y0 = np.array([1, 0])
z0 = np.array([0.01, 0])
k1 = 1.0
k2 = 0.01

forces = np.array([[gravitational_force, x0,   k1],
                   [gravitational_force, y0,   k2],
                   [centrifugal_force,   z0,   k1]])

jacobs = np.array([[gravitational_jacob, x0,   k1],
                   [gravitational_jacob, y0,   k2],
                   [centrifugal_jacob,   None, k1]])

f = f_gen(forces)
df = df_gen(jacobs)

print f(x)
print df(x)
