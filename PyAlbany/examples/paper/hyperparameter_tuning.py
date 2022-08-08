from mpi4py import MPI
import numpy as np
from PyAlbany import Utils
from PyAlbany import FEM_postprocess as fp
import os
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def run_forward(nSweeps, parallelEnv):
    timerNames = ["PyAlbany: Perform Solve"]

    nTimers = len(timerNames)

    filename = "input.yaml"

    parameter = Utils.createParameterList(
        filename, parallelEnv
    )
    parameter.sublist('Piro').sublist('NOX').sublist('Direction').sublist('Newton').sublist('Stratimikos Linear Solver').sublist('Stratimikos').sublist('Preconditioner Types').sublist('Ifpack2').sublist('Ifpack2 Settings').set('relaxation: sweeps', nSweeps)

    timers = Utils.createTimers(timerNames)
    problem = Utils.createAlbanyProblem(parameter, parallelEnv)
    timers[0].start()
    problem.performSolve()
    timers[0].stop()

    return timers[0].totalElapsedTime()


def main(parallelEnv):
    myGlobalRank = MPI.COMM_WORLD.rank

    nMaxSweeps = 300

    timers_sec = np.zeros((nMaxSweeps))

    for nSweeps in range(1, nMaxSweeps+1):
        timers_sec[nSweeps-1] = run_forward(nSweeps, parallelEnv)

    if myGlobalRank==0:
        fig = plt.figure(figsize=(10,6))
        plt.plot(np.arange(1, nMaxSweeps+1), timers_sec)
        plt.savefig('nsweeps.jpeg', dpi=800)

if __name__ == "__main__":
    parallelEnv = Utils.createDefaultParallelEnv()
    main(parallelEnv)
