from PyTrilinos import Tpetra
from PyTrilinos import Teuchos
from Albany_Pybind11 import *

import unittest
import numpy as np
try:
    from PyAlbany import Utils
except:
    import Utils
import os

class TestIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comm = Albany_Pybind11.getDefaultComm()
        cls.parallelEnv = Utils.createDefaultParallelEnv(cls.comm)

    def test_write_distributed_npy(self):
        cls = self.__class__
        rank = cls.comm.getRank()
        nproc = cls.comm.getSize()
        if nproc > 1:
            mvector_filename = 'out_mvector_write_test_' + str(nproc)
        else:
            mvector_filename ='out_mvector_write_test'

        file_dir = os.path.dirname(__file__)

        filename = 'input.yaml'
        '''
        problem = Utils.createAlbanyProblem(file_dir+'/'+filename, cls.parallelEnv)

        n_cols = 4
        parameter_map = problem.getParameterMap(0)
        mvector = Tpetra.MultiVector(parameter_map, n_cols, dtype="d")

        mvector[0,:] = 1.*(rank+1)
        mvector[1,:] = -1.*(rank+1)
        mvector[2,:] = 3.26*(rank+1)
        mvector[3,:] = -3.1*(rank+1)

        Utils.writeMVector(file_dir+'/'+mvector_filename, mvector, distributedFile = True, useBinary = True)
        '''

    @classmethod
    def tearDownClass(cls):
        cls.parallelEnv = None
        cls.comm = None

if __name__ == '__main__':
    unittest.main()