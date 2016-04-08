# coding=utf8

import numpy as np

k1 = 1.0
k2 = 0.01
x0 = np.array([0,0])
y0 = np.array([1,0])
z0 = np.array([0.01,0])






denomin_gravit_force = lambda x,x0: ((x[0] - x0[0])**2 + (x[1] - x0[1])**2)**(3.0/2.0)


elastic_force = lambda dx: -k * dx
centrifugal_force = lambda x,x0,k: k * (x - x0)
gravitational_force = lambda x,x0,k: -k * (x - x0) / denomin_gravit_force(x,x0)


elastic_jacob = lambda k: np.array([[-k,0],
                          [0, -k]])

centrifugal_jacob =lambda k: np.array([[k,0],
                              [0, k]])



gravitational_jacob = lambda x,x0,k: np.array([[-k*(denomin_gravit_force(x,x0)-((x[0]-x0[0])**2)*3*(((x[0] - x0[0])**2) + ((x[1] - x0[1])**2))**(1.0/2.0))/(denomin_gravit_force(x,x0)**2) , -k*(-(x[0]-x0[0])*3*(x[1]-x0[1])*(((x[0] - x0[0])**2) + ((x[1] - x0[1])**2))**(1.0/2.0))/(denomin_gravit_force(x,x0)**2)],
                                               [-k*(-(x[1]-x0[1])*3*(x[0]-x0[0])*(((x[0] - x0[0])**2) + ((x[1] - x0[1])**2))**(1.0/2.0))/(denomin_gravit_force(x,x0)**2) , -k*(denomin_gravit_force(x,x0)-((x[1]-x0[1])**2)*3*(((x[0] - x0[0])**2) + ((x[1] - x0[1])**2))**(1.0/2.0))/(denomin_gravit_force(x,x0)**2)]])

x = np.array([1.5, 0])



#print elastic_force(x)


print centrifugal_force(x,z0,k1) + gravitational_force(x,x0,k1) + gravitational_force(x,y0,k2)
print centrifugal_jacob(k1) + gravitational_jacob(x,x0,k1) + gravitational_jacob(x,y0,k2)
