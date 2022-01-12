from PyTrilinos import Tpetra
from PyTrilinos import Teuchos

from mpi4py import MPI
import numpy as np
from PyAlbany import Utils
from PyAlbany import ExtremeEvent as ee
import os
import sys
from PyAlbany import wpyalbany as wpa

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    printPlot = True
except:
    printPlot = False

def main(parallelEnv):
    comm = MPI.COMM_WORLD
    myGlobalRank = comm.rank

    x, y, min_x, min_y, max_x, max_y = ee.read_mesh_coordinates('steady2d.exo')

    filename = "thermal_steady_distributed.yaml"

    n_KLTerms = 10
    kl = wpa.KLExpention(2)
    print(min_x)
    kl.setLowerBound(0, min_x)
    kl.setLowerBound(1, min_y)
    kl.setUpperBound(0, max_x)
    kl.setUpperBound(1, max_y)
    kl.setNumberOfKLTerms(n_KLTerms)
    kl.setNumberOfKLTerm(0, n_KLTerms)
    kl.setNumberOfKLTerm(1, n_KLTerms)
    kl.createModes()

    phi = np.zeros((len(x), n_KLTerms))
    kl.getModes(phi, x, y)

    plt.figure()
    plt.plot(x, y, '.')
    plt.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], 'r')
    plt.savefig('nodes.jpeg')

    for i in range(0, n_KLTerms):

        Z = np.reshape(phi[:,i], (11, 11))
        X = np.reshape(x, (11, 11))
        Y = np.reshape(y, (11, 11))

        plt.figure()
        plt.contourf(X,Y,Z)
        plt.savefig('phi_'+str(i)+'.jpeg')

    weights = np.random.normal(0, 1, n_KLTerms)

    Z = weights[0] * phi[:,0]
    for i in range(1, n_KLTerms):
        Z += weights[i] * phi[:,i]

    Z = np.reshape(Z, (11, 11))
    X = np.reshape(x, (11, 11))
    Y = np.reshape(y, (11, 11))

    plt.figure()
    plt.contourf(X,Y,Z)
    plt.savefig('random.jpeg')

    parameter = Utils.createParameterList(
        filename, parallelEnv
    )

    ee.update_parameter_list(parameter, n_KLTerms)
    problem = Utils.createAlbanyProblem(parameter, parallelEnv)

    for i in range(1, n_KLTerms+1):
        parameter_map = problem.getParameterMap(i)
        mode_i = Tpetra.MultiVector(parameter_map, 1, dtype="d")

        mode_i[0,:] = phi[:,i-1]
        problem.setParameter(i, mode_i)
    
    '''
    parameter_map = problem.getParameterMap(0)
    mode_i = Tpetra.MultiVector(parameter_map, 1, dtype="d")
    mode_i[0,:] = 1.
    problem.setParameter(0, mode_i)
    '''

    l_min = 8.
    l_max = 20.
    n_l = 5

    p = 1.

    l = l_min + np.power(np.linspace(0.0, 1.0, n_l), p) * (l_max-l_min)

    theta_star, I_star, F_star, P_star = ee.evaluateThetaStar(l, problem, n_KLTerms)

    np.savetxt('theta_star_steady_distributed.txt', theta_star)
    np.savetxt('I_star_steady_distributed.txt', I_star)
    np.savetxt('P_star_steady_distributed.txt', P_star)
    np.savetxt('F_star_steady_distributed.txt', F_star)

if __name__ == "__main__":
    comm = Teuchos.DefaultComm.getComm()
    parallelEnv = Utils.createDefaultParallelEnv(comm)
    main(parallelEnv)
