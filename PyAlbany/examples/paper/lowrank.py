from mpi4py import MPI
import numpy as np
from PyAlbany import Utils
from PyAlbany.RandomizedCompression import doublePass
from PyAlbany import FEM_postprocess as fp
import os
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



class Hessian:
   def __init__(me, problem, parameterIndex, responseIndex):
       me.problem = problem
       me.parameterIndex = parameterIndex
       me.responseIndex  = responseIndex
       me.Map            = me.problem.getParameterMap(me.parameterIndex)
   def dot(me, x):
       me.problem.setDirections(me.parameterIndex, x)
       me.problem.performSolve()
       return me.problem.getReducedHessian(me.responseIndex, me.parameterIndex) 


def main(parallelEnv):
    myGlobalRank = MPI.COMM_WORLD.rank

    # Create an Albany problem:
    filename = "input.yaml"
    parameter = Utils.createParameterList(filename, parallelEnv)

    problem = Utils.createAlbanyProblem(parameter, parallelEnv)
    problem.performAnalysis()
    problem.performSolve()
     
    parameterIndex = 1
    responseIndex  = 0
    Hess = Hessian(problem, parameterIndex, responseIndex)
    
    k = 100
    eigVals, eigVecs = doublePass(Hess, k, symmetric=True)
    if myGlobalRank == 0:
        fig = plt.figure(figsize=(6,4))
        plt.plot(eigVals)
        plt.ylabel('Eigenvalues of the Hessian')
        plt.xlabel('Eigenvalue index')
        plt.gca().set_xlim([0, k])
        plt.gca().set_ylim([6e-4, 1.1e-3])
        plt.grid(True, which="both")
        fig.tight_layout()
        plt.savefig('hessian_eigenvalues.jpeg', dpi=800)
        plt.close()
    

if __name__ == "__main__":
    parallelEnv = Utils.createDefaultParallelEnv()
    main(parallelEnv)
