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
#include "Albany_Pybind11_Tpetra.hpp"
#include "Albany_Pybind11_Timer.hpp"

#include "Albany_Interface.hpp"

namespace py = pybind11;

PYBIND11_MODULE(Albany_Pybind11, m) {
    m.doc() = "PyAlbany with Pybind11";

    pyalbany_comm(m);
    pyalbany_parallelenv(m);
    pyalbany_parameterlist(m);

    pyalbany_map(m);
    pyalbany_vector(m);
    pyalbany_mvector(m);

    py::class_<RCP_Time>(m, "Time")
        .def(py::init(&createRCPTime))
        .def("totalElapsedTime",[](RCP_Time &m){
            return m->totalElapsedTime();
        })
        .def("name",[](RCP_Time &m){
            return m->name();
        });

    py::class_<RCP_StackedTimer>(m, "RCPStackedTimer")
        .def(py::init())
        .def("accumulatedTime",[](RCP_StackedTimer &m, const std::string name){
            return m->accumulatedTime(name);
        })
        .def("baseTimerAccumulatedTime",[](RCP_StackedTimer &m, const std::string name){
            return m->findBaseTimer(name)->accumulatedTime();
        });

    py::class_<PyAlbany::PyProblem>(m, "PyProblem")
        .def(py::init<std::string, Teuchos::RCP<PyAlbany::PyParallelEnv>>())
        .def(py::init<Teuchos::RCP<Teuchos::ParameterList>, Teuchos::RCP<PyAlbany::PyParallelEnv>>())
        .def("performSolve", &PyAlbany::PyProblem::performSolve)
        .def("performAnalysis", &PyAlbany::PyProblem::performAnalysis)
        .def("getResponseMap", &PyAlbany::PyProblem::getResponseMap)
        .def("getStateMap", &PyAlbany::PyProblem::getStateMap)
        .def("getParameterMap", &PyAlbany::PyProblem::getParameterMap)
        .def("setDirections", &PyAlbany::PyProblem::setDirections)
        .def("setParameter", &PyAlbany::PyProblem::setParameter)
        .def("getParameter", &PyAlbany::PyProblem::getParameter)
        .def("getResponse", &PyAlbany::PyProblem::getResponse)
        .def("getState", &PyAlbany::PyProblem::getState)
        .def("getSensitivity", &PyAlbany::PyProblem::getSensitivity)
        .def("getReducedHessian", &PyAlbany::PyProblem::getReducedHessian)
        .def("reportTimers", &PyAlbany::PyProblem::reportTimers)
        .def("getCumulativeResponseContribution", &PyAlbany::PyProblem::getCumulativeResponseContribution)
        .def("updateCumulativeResponseContributionWeigth", &PyAlbany::PyProblem::updateCumulativeResponseContributionWeigth)
        .def("updateCumulativeResponseContributionTargetAndExponent", &PyAlbany::PyProblem::updateCumulativeResponseContributionTargetAndExponent)
        .def("getCovarianceMatrix", &PyAlbany::PyProblem::getCovarianceMatrix)
        .def("setCovarianceMatrix", &PyAlbany::PyProblem::setCovarianceMatrix)
        .def("getStackedTimer", &PyAlbany::PyProblem::getStackedTimer);

    m.def("getRankZeroMap", &PyAlbany::getRankZeroMap, "A function which multiplies two numbers");
    m.def("scatterMVector", &PyAlbany::scatterMVector, "A function which multiplies two numbers");
    m.def("gatherMVector", &PyAlbany::gatherMVector, "A function which multiplies two numbers");
    m.def("orthogTpMVecs", &PyAlbany::orthogTpMVecs, "A function which multiplies two numbers");
}
