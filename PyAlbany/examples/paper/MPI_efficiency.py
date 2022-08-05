from mpi4py import MPI
import numpy as np
from PyAlbany import Utils
from PyAlbany import Albany_Pybind11 as wpa
import os
import sys
import argparse

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    printPlot = True
except:
    printPlot = False

def main(parallelEnv):
    comm = MPI.COMM_WORLD
    nMaxProcs = comm.Get_size()
    myGlobalRank = comm.rank

    parser = argparse.ArgumentParser(description='Select the scaling.')
    parser.add_argument('-w', action="store_true", default=False)
    args = parser.parse_args()

    weak_scaling = args.w

    timerNames = ["PyAlbany: Create Albany Problem", 
                "PyAlbany: Set directions",
                "PyAlbany: Perform Solve",
                "PyAlbany: Total"]

    nTimers = len(timerNames)

    # number of times that the test is repeated for a fixed
    # number of MPI processes
    N = 10

    timers_sec = np.zeros((nMaxProcs,nTimers,N))
    mean_timers_sec = np.zeros((nMaxProcs,nTimers))

    efficiency = np.zeros((nMaxProcs,nTimers))

    for nProcs in range(1, nMaxProcs+1):
        newGroup = comm.group.Incl(np.arange(0, nProcs))
        newComm = comm.Create_group(newGroup)

        if myGlobalRank < nProcs:
            parallelEnv.setComm(wpa.getTeuchosComm(newComm))

            for i_test in range(0,N):
                timers = Utils.createTimers(timerNames)
                timers[3].start()
                timers[0].start()

                filename = "input.yaml"
                parameter = Utils.createParameterList(
                    filename, parallelEnv
                )

                if weak_scaling:
                    parameter.sublist("Discretization").set("2D Elements", 40*nProcs)
                problem = Utils.createAlbanyProblem(parameter, parallelEnv)
                timers[0].stop()

                timers[1].start()
                n_directions = 4
                parameter_map = problem.getParameterMap(0)
                directions = Utils.createMultiVector(parameter_map, n_directions)

                directions_view = directions.getLocalViewHost()
                directions_view[:,0] = 1.
                directions_view[:,1] = -1.
                directions_view[:,2] = 3.
                directions_view[:,3] = -3.
                directions.setLocalViewHost(directions_view)

                problem.setDirections(0, directions)
                timers[1].stop()

                timers[2].start()
                problem.performSolve()
                timers[2].stop()
                timers[3].stop()

                if myGlobalRank == 0:
                    for j in range(0, nTimers):
                        timers_sec[nProcs-1,j,i_test] = timers[j].totalElapsedTime()

    if myGlobalRank == 0:
        for i in range(0, nMaxProcs):
            for j in range(0, nTimers):
                mean_timers_sec[i,j] = np.mean(timers_sec[i,j,:])
            efficiency[i,:] = mean_timers_sec[0,:]/(mean_timers_sec[i,:])
            if not weak_scaling:
                efficiency[i,:] /= (i+1)

        print('timers')
        print(mean_timers_sec)

        print('efficiency')
        print(efficiency)
        if printPlot:
            fig = plt.figure(figsize=(10,6))
            plt.plot([1, nMaxProcs+1], [1., 1.], '--')
            for j in range(0, nTimers):
                plt.plot(np.arange(1, nMaxProcs+1), efficiency[:,j], 'o-', label=timerNames[j])
            plt.ylabel('efficiency')
            plt.xlabel('number of MPI processes')
            plt.grid(True)
            plt.legend()
            if weak_scaling:
                plt.savefig('weak_scaling.jpeg', dpi=800)
            else:
                plt.savefig('strong_scaling.jpeg', dpi=800)
            plt.close()

if __name__ == "__main__":
    comm = wpa.getTeuchosComm(MPI.COMM_WORLD)
    parallelEnv = Utils.createDefaultParallelEnv(comm)
    main(parallelEnv)
