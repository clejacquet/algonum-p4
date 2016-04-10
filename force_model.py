# encoding=utf8

import numpy as np


class ForceModel:
    """
    Represents a model with some forces
    """

    forces = None

    def __init__(self):
        self.forces = []

    def add_force(self, force, origin, k):
        """
        Add a force to the model
        :param force: a force type
        :param origin: its origin point
        :param k: its coefficient value
        """
        self.forces.append([force, origin, k])

    def compute_total_force(self):
        """
        Compute the total function force
        :return: total function force
        """
        def f(vec):
            total = np.array([0., 0.])
            for i in range(0, len(self.forces)):
                force = self.forces[i][0].f
                origin = self.forces[i][1]
                k = self.forces[i][2]

                res = force(vec, origin, k)
                total += res
            return total
        return f

    def compute_total_jacobian(self):
        """
        Compute the total jacobian force
        :return: total jacobian force
        """
        def df(vec):
            total = np.array([[0., 0.],
                              [0., 0.]])
            for i in range(0, len(self.forces)):
                jacobian = self.forces[i][0].df
                origin = self.forces[i][1]
                k = self.forces[i][2]

                res = jacobian(vec, origin, k)
                total += res
            return total
        return df
