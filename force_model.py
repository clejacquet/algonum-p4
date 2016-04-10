# encoding=utf8

import numpy as np


class ForceModel:
    """

    """
    forces = None

    def __init__(self):
        """

        :return:
        """
        self.forces = []

    def add_force(self, force, origin, k):
        """

        :param force:
        :param origin:
        :param k:
        :return:
        """
        self.forces.append([force, origin, k])

    def compute_total_force(self):
        """

        :return:
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

        :return:
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
