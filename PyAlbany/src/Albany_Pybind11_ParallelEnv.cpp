//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Pybind11_ParallelEnv.hpp"
#include "Kokkos_Core.hpp"

namespace py = pybind11;

PyParallelEnv createPyParallelEnv(RCP_Teuchos_Comm_PyAlbany _comm, int _num_threads, int _num_numa, int _device_id) {
    return PyAlbany::PyParallelEnv(_comm, _num_threads, _num_numa, _device_id);
}

PyParallelEnv createDefaultKokkosPyParallelEnv(RCP_Teuchos_Comm_PyAlbany _comm) {
    return PyAlbany::PyParallelEnv(_comm, -1, -1, -1);
}

void pyalbany_parallelenv(py::module &m) {
    py::class_<PyParallelEnv, Teuchos::RCP<PyParallelEnv>>(m, "PyParallelEnv")
        .def(py::init(&createPyParallelEnv))
        .def(py::init(&createDefaultKokkosPyParallelEnv))
        .def("getNumThreads", &PyParallelEnv::getNumThreads)
        .def("getNumNuma", &PyParallelEnv::getNumNuma)
        .def("getDeviceID", &PyParallelEnv::getDeviceID)
        .def("getComm", &PyParallelEnv::getComm)
        .def("setComm", &PyParallelEnv::setComm);
}
