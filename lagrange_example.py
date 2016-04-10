# encoding=utf8

import lagrange_model as lm
import lagrange_solver as ls
import forces as f
import numpy as np


def main():
    model = lm.LagrangeModel()

    model.add_force(f.FORCE_LIST['centrifugal'],   np.array([0.01, 0.]), 1.)
    model.add_force(f.FORCE_LIST['gravitational'], np.array([0.,   0.]), 1.)
    model.add_force(f.FORCE_LIST['gravitational'], np.array([1.,   0.]), 0.01)

    solver = ls.LagrangeSolver(model)
    solver.show_solutions([-2., -2., 2., 2.], 10., 10000, 1e-10)

if __name__ == '__main__':
    main()
