# encoding=utf8

import numpy as np
import NRMethod as nr
import forces as f
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
        force_types = {}
        for name, force in f.FORCE_LIST.iteritems():
            force_types[name] = {
                'x': [],
                'y': [],
                'color': force.color
            }

        for force in self.model.forces:
            force_types[force[0].name]['x'].append(force[1][0])
            force_types[force[0].name]['y'].append(force[1][1])

        ax = plt.subplot(111)
        scatters = {}
        for name, force_type in force_types.iteritems():
            scatters[name] = ax.scatter(force_type['x'], force_type['y'], color=force_type['color'])

        solutions = np.array(self.solve(window, step_count, n_max, eps))

        scatters['solutions'] = ax.scatter(solutions[:, 0], solutions[:, 1], color='grey', marker='x')

        plt.legend(scatters.values(),
           scatters.keys(),
           scatterpoints=1,
           loc='lower left',
           ncol=1,
           fontsize=12)

        plt.title('Equilibrium points')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
