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
    filename = "input_dirichletT.yaml"
    parameter = Utils.createParameterList(
        filename, parallelEnv
    )

    problem = Utils.createAlbanyProblem(parameter, parallelEnv)

    #----------------------------------------------
    #
    #      1. Evaluation of the theta star
    #
    #----------------------------------------------

    l_min = 0.1
    l_max = 2
    n_l = 1

    l = np.linspace(l_min, l_max, n_l)

    theta_star = np.zeros((n_l, 2))
    I_star = np.zeros((n_l,))
    F_star = np.zeros((n_l,))

    # Loop over the lambdas
    for i in range(0, n_l):
        #parameter.sublist("Problem").sublist("Response Functions").sublist("Response 0").sublist("Response 1").set("Scaling Coefficient", -l[i])

        problem.performAnalysis()

        para_0 = problem.getParameter(0)
        theta_star[i, :] = para_0.getData()

        problem.performSolve()
        response = problem.getResponse(0)
        #response_2 = problem.getResponse(2)

        I_star[i] = response.getData()[0]
        #F_star[i] = response_2.getData()[0]

    P_star = np.exp(-I_star)

    print(theta_star)
    print(I_star)

    if myGlobalRank == 0:
        if printPlot:
            plt.figure()
            plt.semilogy(F_star,P_star)

            plt.savefig('extreme.jpeg', dpi=800)
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