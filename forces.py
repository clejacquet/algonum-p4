# coding=utf8

import numpy as np


def denomin_gravit_force(x, x0):
    return ((x[0] - x0[0])**2 + (x[1] - x0[1])**2)**(3.0 / 2.0)


# FORCES


def elastic_force(x, x0, k):
    return -k * x


def centrifugal_force(x, x0, k):
    return k * (x - x0)


def gravitational_force(x, x0, k):
    return -k * (x - x0) / denomin_gravit_force(x, x0)


# JACOBIANS


def elastic_jacob(x, x0, k):
    return np.array([[-k, 0],
                     [0, -k]])


def centrifugal_jacob(x, x0, k):
    return np.array([[k, 0],
                     [0, k]])


def gravitational_jacob(x, x0, k):
    return np.array([[-k*(denomin_gravit_force(x,x0)-((x[0]-x0[0])**2)*3*(((x[0] - x0[0])**2) + ((x[1] - x0[1])**2))**(1.0/2.0))/(denomin_gravit_force(x,x0)**2),
                      -k*(-(x[0]-x0[0])*3*(x[1]-x0[1])*(((x[0] - x0[0])**2) + ((x[1] - x0[1])**2))**(1.0/2.0))/(denomin_gravit_force(x,x0)**2)],
                     [-k*(-(x[1]-x0[1])*3*(x[0]-x0[0])*(((x[0] - x0[0])**2) + ((x[1] - x0[1])**2))**(1.0/2.0))/(denomin_gravit_force(x,x0)**2) ,
                      -k*(denomin_gravit_force(x,x0)-((x[1]-x0[1])**2)*3*(((x[0] - x0[0])**2) + ((x[1] - x0[1])**2))**(1.0/2.0))/(denomin_gravit_force(x,x0)**2)]])


class Force:
    """
    Represents a force, by its function and its jacobian function
    """

    f = None
    df = None
    name = None
    color = None

    def __init__(self, f, df, name, color):
        self.f = f
        self.df = df
        self.name = name
        self.color = color


""" A list of every types of force which are handled """
FORCE_LIST = {
    'elastic':       Force(elastic_force,       elastic_jacob,       'elastic',       'blue'),
    'centrifugal':   Force(centrifugal_force,   centrifugal_jacob,   'centrifugal',   'green'),
    'gravitational': Force(gravitational_force, gravitational_jacob, 'gravitational', 'red')
}