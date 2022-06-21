#include "Teuchos_RCP.hpp"
#include "Teuchos_DefaultComm.hpp"

#include <pybind11/pybind11.h>

using Teuchos_Comm_PyAlbany = Teuchos::Comm<int>;
using RCP_Teuchos_Comm_PyAlbany = Teuchos::RCP<const Teuchos::Comm<int> >;

namespace py = pybind11;

RCP_Teuchos_Comm_PyAlbany
getDefaultComm () {
    //return Teuchos::DefaultComm<int>::getComm();
    return Teuchos::rcp (new Teuchos::SerialComm<int> ());
}


PYBIND11_MODULE(Albany_Pybind11, m) {
    py::class_<RCP_Teuchos_Comm_PyAlbany>(m, "RCP_Comm")
        .def(py::init<>())
        .def("getRank", [](RCP_Teuchos_Comm_PyAlbany &m) {
            return m->getRank();
        })
        .def("getSize", [](RCP_Teuchos_Comm_PyAlbany &m) {
            return m->getSize();
        });

    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("getDefaultComm", &getDefaultComm, "A function which multiplies two numbers");
}
