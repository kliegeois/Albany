#include "Teuchos_RCP.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_DefaultComm.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#if PY_VERSION_HEX >= 0x03000000

#define PyClass_Check(obj) PyObject_IsInstance(obj, (PyObject *)&PyType_Type)
#define PyInt_Check(x) PyLong_Check(x)
#define PyInt_AsLong(x) PyLong_AsLong(x)
#define PyInt_FromLong(x) PyLong_FromLong(x)
#define PyInt_FromSize_t(x) PyLong_FromSize_t(x)
#define PyString_Check(name) PyBytes_Check(name)
#define PyString_FromString(x) PyUnicode_FromString(x)
#define PyString_FromStringAndSize(x,s) PyUnicode_FromStringAndSize(x,s)
#define PyString_Format(fmt, args)  PyUnicode_Format(fmt, args)
#define PyString_AsString(str) PyBytes_AsString(str)
#define PyString_Size(str) PyBytes_Size(str)    
#define PyString_InternFromString(key) PyUnicode_InternFromString(key)
#define Py_TPFLAGS_HAVE_CLASS Py_TPFLAGS_BASETYPE
#define PyString_AS_STRING(x) PyUnicode_AS_STRING(x)
#define PyObject_Compare(x, y) (1-PyObject_RichCompareBool(x, y, Py_EQ))
#define _PyLong_FromSsize_t(x) PyLong_FromSsize_t(x)
#define convertPyStringToChar(pyobj) PyBytes_AsString(PyUnicode_AsASCIIString(pyobj))
#else
#define convertPyStringToChar(pyobj) PyString_AsString(pyobj)
#endif

#include "Albany_Pybind11_Comm.hpp"
#include "Albany_Pybind11_ParallelEnv.hpp"
#include "Albany_Pybind11_ParameterList.hpp"

namespace py = pybind11;

PYBIND11_MODULE(Albany_Pybind11, m) {
    m.doc() = "PyAlbany with Pybind11";

    py::class_<RCP_Teuchos_Comm_PyAlbany>(m, "PyComm")
        .def(py::init<>())
        .def("getRank", [](RCP_Teuchos_Comm_PyAlbany &m) {
            return m->getRank();
        })
        .def("getSize", [](RCP_Teuchos_Comm_PyAlbany &m) {
            return m->getSize();
        });

    m.def("getDefaultComm", &getDefaultComm, "A function which multiplies two numbers");
    m.def("getTeuchosComm", &getTeuchosComm, "A function which multiplies two numbers");
    m.def("finalize", &finalize, "A function which multiplies two numbers");

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

    py::class_<PyParameterList>(m, "PyParameterList")
        .def(py::init(&createPyParameterList))
        .def("sublist", [](PyParameterList &m, const std::string &name) {
            return m.sublist(name);
        })
        .def("setSublist", [](PyParameterList &m, const std::string &name, PyParameterList &sub) {
            m.set(name, sub);
        })
        .def("setSublist", [](PyParameterList &m, const std::string &name, RCP_PyParameterList &sub) {
            m.set(name, *sub);
        })
        .def("isParameter", [](PyParameterList &m, const std::string &name) {
            return m.isParameter(name);
        })
        .def("get", [](PyParameterList &m, const std::string &name) {
            if (m.isParameter(name)) {
                return getPythonParameter(m, name);
            }
            return py::cast("Invalid parameter name");
        })
        .def("set", [](PyParameterList &m, const std::string &name, py::object value) {
            if (!setPythonParameter(m,name,value))
                PyErr_SetString(PyExc_TypeError, "ParameterList value type not supported");
        });

    py::class_<RCP_PyParameterList>(m, "RCPPyParameterList")
        .def(py::init(&createRCPPyParameterList))
        .def("sublist", [](RCP_PyParameterList &m, const std::string &name) {
            return m->sublist(name);
        })
        .def("setSublist", [](RCP_PyParameterList &m, const std::string &name, PyParameterList &sub) {
            m->set(name, sub);
        })
        .def("setSublist", [](RCP_PyParameterList &m, const std::string &name, RCP_PyParameterList &sub) {
            m->set(name, *sub);
        })
        .def("isParameter", [](RCP_PyParameterList &m, const std::string &name) {
            return m->isParameter(name);
        })
        .def("get", [](RCP_PyParameterList &m, const std::string &name) {
            if (m->isParameter(name)) {
                return getPythonParameter(*m, name);
            }
            return py::cast("Invalid parameter name");
        })
        .def("set", [](RCP_PyParameterList &m, const std::string &name, py::object value) {
            if (!setPythonParameter(*m,name,value))
                PyErr_SetString(PyExc_TypeError, "ParameterList value type not supported");
        });
    m.def("getParameterList", &getParameterList, "A function which multiplies two numbers");
}
