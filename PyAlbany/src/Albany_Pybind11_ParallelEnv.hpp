#ifndef PYALBANY_PARALLELENV_H
#define PYALBANY_PARALLELENV_H

#include "Albany_Pybind11_Comm.hpp"
#include "Albany_Interface.hpp"

using RCP_PyParallelEnv = Teuchos::RCP<PyAlbany::PyParallelEnv>;

RCP_PyParallelEnv createPyParallelEnv(RCP_Teuchos_Comm_PyAlbany _comm, int _num_threads = -1, int _num_numa = -1, int _device_id = -1);
RCP_PyParallelEnv createDefaultKokkosPyParallelEnv(RCP_Teuchos_Comm_PyAlbany _comm);

void pyalbany_parallelenv(pybind11::module &m);

#endif
