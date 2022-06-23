#include "Teuchos_RCP.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_DefaultComm.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "Albany_Pybind11_Comm.hpp"
#include "Albany_Pybind11_ParallelEnv.hpp"

namespace py = pybind11;

#define PyString_Check(name) PyBytes_Check(name)

PYBIND11_MODULE(Albany_Pybind11, m) {
    py::class_<RCP_Teuchos_Comm_PyAlbany>(m, "PyComm")
        .def(py::init<>())
        .def("getRank", [](RCP_Teuchos_Comm_PyAlbany &m) {
            return m->getRank();
        })
        .def("getSize", [](RCP_Teuchos_Comm_PyAlbany &m) {
            return m->getSize();
        });

    py::class_<PyParallelEnv>(m, "PyParallelEnv")
        .def(py::init<RCP_Teuchos_Comm_PyAlbany>())
        .def(py::init<RCP_Teuchos_Comm_PyAlbany, int, int, int>())
        .def_readwrite("comm", &PyParallelEnv::comm)
        .def_readonly("num_threads", &PyParallelEnv::num_threads)
        .def_readonly("num_numa", &PyParallelEnv::num_numa)
        .def_readonly("device_id", &PyParallelEnv::device_id);

    m.doc() = "pybind11 example plugin";
    m.def("getDefaultComm", &getDefaultComm, "A function which multiplies two numbers");
    m.def("getTeuchosComm", &getTeuchosComm, "A function which multiplies two numbers");
    m.def("finalize", &finalize, "A function which multiplies two numbers");
}
