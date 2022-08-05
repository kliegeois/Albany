from mpi4py import MPI
import numpy as np
from PyAlbany import Utils
from PyAlbany import FEM_postprocess
import os
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def main(parallelEnv):
    myGlobalRank = MPI.COMM_WORLD.rank

    # Create an Albany problem:
    filename = "input.yaml"
    parameter = Utils.createParameterList(
        filename, parallelEnv
    )

    problem = Utils.createAlbanyProblem(parameter, parallelEnv)
    problem.performSolve()

    if myGlobalRank==0:
        x, y, sol, elements, triangulation = FEM_postprocess.readExodus("steady2d.exo", ['solution', 'thermal_conductivity', 'thermal_conductivity_sensitivity'], MPI.COMM_WORLD.Get_size())

        plt.figure()
        FEM_postprocess.plot_fem_mesh(x, y, elements)
        plt.tricontourf(triangulation, sol[0,:])
        plt.colorbar()
        plt.axis('equal')
        plt.savefig('sol.jpeg', dpi=800)

        plt.figure()
        FEM_postprocess.plot_fem_mesh(x, y, elements)
        plt.tricontourf(triangulation, sol[1,:])
        plt.colorbar()
        plt.axis('equal')
        plt.savefig('thermal_conductivity.jpeg', dpi=800)

        plt.figure()
        FEM_postprocess.plot_fem_mesh(x, y, elements)
        plt.tricontourf(triangulation, sol[2,:])
        plt.colorbar()
        plt.axis('equal')
        plt.savefig('thermal_conductivity_sensitivity.jpeg', dpi=800)

if __name__ == "__main__":
    parallelEnv = Utils.createDefaultParallelEnv()
    main(parallelEnv)
