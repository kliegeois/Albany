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

    py::class_<RCP_PyParallelEnv>(m, "PyParallelEnv")
        .def(py::init(&createPyParallelEnv))
        .def(py::init(&createDefaultKokkosPyParallelEnv))
        .def("getNumThreads", [](RCP_PyParallelEnv &m) {
            return m->num_threads;
        })
        .def("getNumNuma", [](RCP_PyParallelEnv &m) {
            return m->num_numa;
        })
        .def("getDeviceID", [](RCP_PyParallelEnv &m) {
            return m->device_id;
        })
        .def("getComm", [](RCP_PyParallelEnv &m) {
            return m->comm;
        })
        .def("setComm", [](RCP_PyParallelEnv &m, RCP_Teuchos_Comm_PyAlbany &comm) {
            m->comm = comm;
        });

    m.doc() = "pybind11 example plugin";
    m.def("getDefaultComm", &getDefaultComm, "A function which multiplies two numbers");
    m.def("getTeuchosComm", &getTeuchosComm, "A function which multiplies two numbers");
    m.def("finalize", &finalize, "A function which multiplies two numbers");
}
