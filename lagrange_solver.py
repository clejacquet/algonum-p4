# encoding=utf8

import numpy as np
import NRMethod as nr
import matplotlib.pyplot as plt


class LagrangeSolver:
    def __init__(self, model):
        self.model = model

    def solve(self, window, step_count, n_max, eps):
        f = self.model.compute_total_force()
        df = self.model.compute_total_jacobian()

        solutions = []

        for i in np.arange(window[0], window[2], (window[2] - window[0]) / step_count):
            for j in np.arange(window[1], window[3], (window[3] - window[1]) / step_count):
                s = nr.Newton_BT(f, df, np.array([i, j]), n_max, eps)
                if len(filter(lambda x: np.allclose(s, x), solutions)) == 0:
                    solutions.append(s)

        return solutions

    def show_solutions(self, window, step_count, n_max, eps):
        x = []
        y = []
        colors = []

        for force in self.model.forces:
            x.append(force[1][0])
            y.append(force[1][1])
            colors.append(force[0].color)

        solutions = self.solve(window, step_count, n_max, eps)

        for solution in solutions:
            x.append(solution[0])
            y.append(solution[1])
            colors.append('grey')

        plt.scatter(x, y, color=colors)
        plt.title('Nuage de points avec Matplotlib')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
