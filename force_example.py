# encoding=utf8

import force_model as fm
import force_equilibrium_solver as fes
import forces as f
import numpy as np


def main():
    model = fm.ForceModel()

    model.add_force(f.FORCE_LIST['elastic'],   np.array([0., 0.]), 1.)
    model.add_force(f.FORCE_LIST['centrifugal'], np.array([0.5,   0.2]), 1.)
    model.add_force(f.FORCE_LIST['centrifugal'], np.array([-0.5,   0.2]), 1.)
    model.add_force(f.FORCE_LIST['gravitational'], np.array([-0.5,   0.3]), 0.01)

    solver = fes.ForceEquilibriumSolver(model)
    solver.show_solutions([-2., -2., 2., 2.], 10., 10000, 1e-10)

if __name__ == '__main__':
    main()
