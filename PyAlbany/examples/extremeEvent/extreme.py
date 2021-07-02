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
    printPlot = True
except:
    printPlot = False

def main(parallelEnv):
    comm = MPI.COMM_WORLD
    myGlobalRank = comm.rank

    # Create an Albany problem:

    n_params = 2
    if n_params==1:
        filename = "input_dirichletT_1.yaml"
    else:
        filename = "input_dirichletT_2.yaml"

    #----------------------------------------------
    #
    #      1. Evaluation of the theta star
    #
    #----------------------------------------------

    l_min = 0.1
    l_max = 3.5
    n_l = 30

    l = np.linspace(l_min, l_max, n_l)

    theta_star = np.zeros((n_l, n_params))
    I_star = np.zeros((n_l,))
    F_star = np.zeros((n_l,))

    # Loop over the lambdas
    for i in range(0, n_l):
        parameter = Utils.createParameterList(
            filename, parallelEnv
        )

        parameter.sublist("Problem").sublist("Response Functions").sublist("Response 0").set("Scaling Coefficient 1", -l[i])
        problem = Utils.createAlbanyProblem(parameter, parallelEnv)

        problem.performAnalysis()

        for j in range(0, n_params):
            para = problem.getParameter(j)
            theta_star[i, j] = para.getData()

        problem.performSolve()

        data = np.loadtxt('CumulativeScalarResponseFunction.txt')
        I_star[i] = data[0]
        F_star[i] = data[1]

    P_star = np.exp(-I_star)

    print(theta_star)
    print(I_star)

    if myGlobalRank == 0:
        if printPlot:
            plt.figure()
            plt.semilogy(F_star,P_star,'*-')

            plt.savefig('extreme.jpeg', dpi=800)
            plt.close()

            if n_params==2:
                plt.figure()
                plt.plot(theta_star[:,0],theta_star[:,1],'*-')

                plt.savefig('theta_star.jpeg', dpi=800)
                plt.close()
    #----------------------------------------------
    #
    #   2. Evaluation of the prefactor using IS
    #
    #----------------------------------------------

if __name__ == "__main__":
    comm = Teuchos.DefaultComm.getComm()
    parallelEnv = Utils.createDefaultParallelEnv(comm)
    main(parallelEnv)
