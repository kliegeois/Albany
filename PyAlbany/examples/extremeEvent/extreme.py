from PyTrilinos import Tpetra
from PyTrilinos import Teuchos

from mpi4py import MPI
import numpy as np
from PyAlbany import Utils
import os
import sys

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    printPlot = True
except:
    printPlot = False


def evaluate_responses(X, Y, problem, recompute=False):
    if not recompute and os.path.isfile('Z1.txt'):
        Z1 = np.loadtxt('Z1.txt')
        Z2 = np.loadtxt('Z2.txt')
    else:
        comm = MPI.COMM_WORLD
        myGlobalRank = comm.rank

        parameter_map_0 = problem.getParameterMap(0)
        parameter_0 = Tpetra.Vector(parameter_map_0, dtype="d")

        parameter_map_1 = problem.getParameterMap(1)
        parameter_1 = Tpetra.Vector(parameter_map_1, dtype="d")

        n_x = len(X)
        n_y = len(Y)
        Z1 = np.zeros((n_y, n_x))
        Z2 = np.zeros((n_y, n_x))

        for i in range(n_x):
            parameter_0[0] = X[i]
            problem.setParameter(0, parameter_0)
            for j in range(n_y):
                parameter_1[0] = Y[j]
                problem.setParameter(1, parameter_1)

                problem.performSolve()

                Z1[j, i] = problem.getCumulativeResponseContribution(0, 0)
                Z2[j, i] = problem.getCumulativeResponseContribution(0, 1)

        np.savetxt('Z1.txt', Z1)
        np.savetxt('Z2.txt', Z2)
    return Z1, Z2


def main(parallelEnv):
    comm = MPI.COMM_WORLD
    myGlobalRank = comm.rank

    # Create an Albany problem:

    n_params = 2
    if n_params == 1:
        filename = "input_dirichletT_1.yaml"
    else:
        filename = "input_dirichletT_2.yaml"

    # ----------------------------------------------
    #
    #      1. Evaluation of the theta star
    #
    # ----------------------------------------------

    l_min = 0.1
    l_max = 2.5
    n_l = 20

    l = np.linspace(l_min, l_max, n_l)

    theta_star = np.zeros((n_l, n_params))
    I_star = np.zeros((n_l,))
    F_star = np.zeros((n_l,))

    parameter = Utils.createParameterList(
        filename, parallelEnv
    )
    problem = Utils.createAlbanyProblem(parameter, parallelEnv)

    # Loop over the lambdas
    for i in range(0, n_l):
        problem.updateCumulativeResponseContributionWeigth(0, 1, -l[i])
        problem.performAnalysis()

        for j in range(0, n_params):
            para = problem.getParameter(j)
            theta_star[i, j] = para.getData()

        problem.performSolve()

        I_star[i] = problem.getCumulativeResponseContribution(0, 0)
        F_star[i] = problem.getCumulativeResponseContribution(0, 1)

    P_star = np.exp(-I_star)

    print(theta_star)
    print(I_star)

    if n_params == 2:
        X = np.arange(-5, 5, 0.2)
        Y = np.arange(-5, 5, 0.25)

        Z1, Z2 = evaluate_responses(X, Y, problem)

        X, Y = np.meshgrid(X, Y)

    if myGlobalRank == 0:
        if printPlot:
            plt.figure()
            plt.semilogy(F_star, P_star, '*-')

            plt.savefig('extreme.jpeg', dpi=800)
            plt.close()

            if n_params == 2:
                plt.figure()
                plt.plot(theta_star[:, 0], theta_star[:, 1], '*-')
                plt.contour(X, Y, Z1, levels=I_star, colors='g')
                plt.contour(X, Y, Z2, levels=F_star, colors='r')
                plt.savefig('theta_star.jpeg', dpi=800)
                plt.close()

                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.plot_surface(X, Y, Z1)

                plt.savefig('Z1.jpeg', dpi=800)
                plt.close()

                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.plot_surface(X, Y, Z2)

                plt.savefig('Z2.jpeg', dpi=800)
                plt.close()

    # ----------------------------------------------
    #
    #   2. Evaluation of the prefactor using IS
    #
    # ----------------------------------------------


if __name__ == "__main__":
    comm = Teuchos.DefaultComm.getComm()
    parallelEnv = Utils.createDefaultParallelEnv(comm)
    main(parallelEnv)
