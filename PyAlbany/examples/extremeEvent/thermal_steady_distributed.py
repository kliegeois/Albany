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
    import exomerge
except:
    import exomerge2 as exomerge

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    printPlot = True
except:
    printPlot = False


def read_mesh_coordinates(filename):
    model = exomerge.import_model(filename)
    positions = np.array(model.nodes)
    x = positions[:,0]
    y = positions[:,1]

    min_x = np.min(x)
    min_y = np.min(y)
    max_x = np.max(x)
    max_y = np.max(y)

    return x, y, min_x, min_y, max_x, max_y

def main(parallelEnv):
    comm = MPI.COMM_WORLD
    myGlobalRank = comm.rank

    x, y, min_x, min_y, max_x, max_y = read_mesh_coordinates('steady2d.exo')
    filename = "thermal_steady_distributed.yaml"

    problem = Utils.createAlbanyProblem(filename, parallelEnv)

    parameter_map = problem.getParameterMap(0)
    x_vector = Tpetra.MultiVector(parameter_map, 1, dtype="d")
    y_vector = Tpetra.MultiVector(parameter_map, 1, dtype="d")

    print(x.shape)
    print(x_vector.shape)
    x_vector[0,:] = x
    y_vector[0,:] = y

    kl = wpa.KLExpention(2)
    print(min_x)
    kl.setLowerBound(0, min_x)
    kl.setLowerBound(1, min_y)
    kl.setUpperBound(0, max_x)
    kl.setUpperBound(1, max_y)
    kl.setNumberOfKLTerms(10)
    kl.setNumberOfKLTerm(0, 10)
    kl.setNumberOfKLTerm(1, 10)
    kl.createModes()


    plt.plot(x_vector[0,:], y_vector[0,:], '.')
    plt.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], 'r')
    plt.savefig('nodes.jpeg')

    for i in range(0, 10):
        phi_i = kl.getMode(i, x_vector, y_vector)

        print(phi_i)

        Z = np.reshape(phi_i, (11, 11))
        X = np.reshape(x, (11, 11))
        Y = np.reshape(y, (11, 11))

        plt.contourf(X,Y,Z)
        plt.plot(x_vector[0,:], y_vector[0,:], '.')
        plt.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], 'r')
        plt.savefig('phi_'+str(i)+'.jpeg')

if __name__ == "__main__":
    comm = Teuchos.DefaultComm.getComm()
    parallelEnv = Utils.createDefaultParallelEnv(comm)
    main(parallelEnv)
