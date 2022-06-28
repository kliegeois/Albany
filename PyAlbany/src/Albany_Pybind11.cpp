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

    py::enum_<Teuchos::EReductionType>(m, "EReductionType")
        .value("REDUCE_SUM", Teuchos::REDUCE_SUM)
        .value("REDUCE_MIN", Teuchos::REDUCE_MIN)
        .value("REDUCE_MAX", Teuchos::REDUCE_MAX)
        .value("REDUCE_AND", Teuchos::REDUCE_AND)
        .value("REDUCE_BOR", Teuchos::REDUCE_BOR)
        .export_values();

    py::class_<RCP_Teuchos_Comm_PyAlbany>(m, "PyComm")
        .def(py::init<>())
        .def("getRank", [](RCP_Teuchos_Comm_PyAlbany &m) {
            return m->getRank();
        })
        .def("getSize", [](RCP_Teuchos_Comm_PyAlbany &m) {
            return m->getSize();
        })
        .def("reduceAll", [](RCP_Teuchos_Comm_PyAlbany &m, Teuchos::EReductionType reductOp, PyObject * sendObj) {
            return reduceAll(m, reductOp, sendObj);
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

    py::class_<RCP_PyParameterList>(m, "RCPPyParameterList")
        .def(py::init(&createRCPPyParameterList))
        .def("sublist", [](RCP_PyParameterList &m, const std::string &name) {
            if (m->isSublist(name))
                return py::cast(sublist(m,name));
            return py::cast("Invalid sublist name");
        }, py::return_value_policy::reference)
        .def("print", [](RCP_PyParameterList &m) {
            m->print();
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

    py::class_<RCP_PyMap>(m, "RCPPyMap")
        .def(py::init(&createRCPPyMapEmpty))
        .def(py::init(&createRCPPyMap))
        .def(py::init(&createRCPPyMapFromView))
        .def("isOneToOne", [](RCP_PyMap &m) {
            return m->isOneToOne();
        })
        .def("getIndexBase", [](RCP_PyMap &m) {
            return m->getIndexBase();
        })
        .def("getMinLocalIndex", [](RCP_PyMap &m) {
            return m->getMinLocalIndex();
        })
        .def("getMaxLocalIndex", [](RCP_PyMap &m) {
            return m->getMaxLocalIndex();
        })
        .def("getMinGlobalIndex", [](RCP_PyMap &m) {
            return m->getMinGlobalIndex();
        })
        .def("getMaxGlobalIndex", [](RCP_PyMap &m) {
            return m->getMaxGlobalIndex();
        })
        .def("getMinAllGlobalIndex", [](RCP_PyMap &m) {
            return m->getMinAllGlobalIndex();
        })
        .def("getMaxAllGlobalIndex", [](RCP_PyMap &m) {
            return m->getMaxAllGlobalIndex();
        })
        .def("getLocalNumElements", [](RCP_PyMap &m) {
            return m->getLocalNumElements();
        })
        .def("getGlobalNumElements", [](RCP_PyMap &m) {
            return m->getGlobalNumElements();
        })
        .def("getLocalElement", [](RCP_PyMap &m, const Tpetra_GO i) {
            return m->getLocalElement(i);
        })
        .def("getGlobalElement", [](RCP_PyMap &m, const Tpetra_LO i) {
            return m->getGlobalElement(i);
        })
        .def("isNodeGlobalElement", [](RCP_PyMap &m, const Tpetra_GO i) {
            return m->isNodeGlobalElement(i);
        })
        .def("isNodeLocalElement", [](RCP_PyMap &m, const Tpetra_LO i) {
            return m->isNodeLocalElement(i);
        })
        .def("isUniform", [](RCP_PyMap &m) {
            return m->isUniform();
        })
        .def("isContiguous", [](RCP_PyMap &m) {
            return m->isContiguous();
        })
        .def("isDistributed", [](RCP_PyMap &m) {
            return m->isDistributed();
        })
        .def("isCompatible", [](RCP_PyMap &m, RCP_PyMap &m2) {
            return m->isCompatible(*m2);
        })
        .def("isSameAs", [](RCP_PyMap &m, RCP_PyMap &m2) {
            return m->isSameAs(*m2);
        })
        .def("locallySameAs", [](RCP_PyMap &m, RCP_PyMap &m2) {
            return m->locallySameAs(*m2);
        })
        .def("getComm", [](RCP_PyMap &m) {
            return m->getComm();
        });

    py::class_<RCP_ConstPyMap>(m, "RCPConstPyMap")
        .def(py::init(&createRCPPyMapEmpty))
        .def(py::init(&createRCPPyMap))
        .def(py::init(&createRCPPyMapFromView))
        .def("isOneToOne", [](RCP_ConstPyMap &m) {
            return m->isOneToOne();
        })
        .def("getIndexBase", [](RCP_ConstPyMap &m) {
            return m->getIndexBase();
        })
        .def("getMinLocalIndex", [](RCP_ConstPyMap &m) {
            return m->getMinLocalIndex();
        })
        .def("getMaxLocalIndex", [](RCP_ConstPyMap &m) {
            return m->getMaxLocalIndex();
        })
        .def("getMinGlobalIndex", [](RCP_ConstPyMap &m) {
            return m->getMinGlobalIndex();
        })
        .def("getMaxGlobalIndex", [](RCP_ConstPyMap &m) {
            return m->getMaxGlobalIndex();
        })
        .def("getMinAllGlobalIndex", [](RCP_ConstPyMap &m) {
            return m->getMinAllGlobalIndex();
        })
        .def("getMaxAllGlobalIndex", [](RCP_ConstPyMap &m) {
            return m->getMaxAllGlobalIndex();
        })
        .def("getLocalNumElements", [](RCP_ConstPyMap &m) {
            return m->getLocalNumElements();
        })
        .def("getGlobalNumElements", [](RCP_ConstPyMap &m) {
            return m->getGlobalNumElements();
        })
        .def("getLocalElement", [](RCP_ConstPyMap &m, const Tpetra_GO i) {
            return m->getLocalElement(i);
        })
        .def("getGlobalElement", [](RCP_ConstPyMap &m, const Tpetra_LO i) {
            return m->getGlobalElement(i);
        })
        .def("isNodeGlobalElement", [](RCP_ConstPyMap &m, const Tpetra_GO i) {
            return m->isNodeGlobalElement(i);
        })
        .def("isNodeLocalElement", [](RCP_ConstPyMap &m, const Tpetra_LO i) {
            return m->isNodeLocalElement(i);
        })
        .def("isUniform", [](RCP_ConstPyMap &m) {
            return m->isUniform();
        })
        .def("isContiguous", [](RCP_ConstPyMap &m) {
            return m->isContiguous();
        })
        .def("isDistributed", [](RCP_ConstPyMap &m) {
            return m->isDistributed();
        })
        .def("isCompatible", [](RCP_ConstPyMap &m, RCP_ConstPyMap &m2) {
            return m->isCompatible(*m2);
        })
        .def("isSameAs", [](RCP_ConstPyMap &m, RCP_ConstPyMap &m2) {
            return m->isSameAs(*m2);
        })
        .def("locallySameAs", [](RCP_ConstPyMap &m, RCP_ConstPyMap &m2) {
            return m->locallySameAs(*m2);
        })
        .def("getComm", [](RCP_ConstPyMap &m) {
            return m->getComm();
        });

    py::class_<RCP_PyVector>(m, "RCPPyVector")
        .def(py::init(&createRCPPyVector1))
        .def(py::init(&createRCPPyVector2))
        .def(py::init(&createRCPPyVectorEmpty))
        .def("putScalar",[](RCP_PyVector &m, ST val) {
            m->putScalar(val);
        })
        .def("getLocalViewHost",[](RCP_PyVector &m){
            return getLocalViewHost(m);
        })
        .def("setLocalViewHost",[](RCP_PyVector &m, py::array_t<ST> input){
            return setLocalViewHost(m, input);
        })
        .def("getMap",[](RCP_PyVector &m){
            return m->getMap();
        })
        .def("dot",[](RCP_PyVector &m, RCP_PyVector &m2){
            return m->dot(*m2);
        });

    py::class_<RCP_PyMultiVector>(m, "RCPPyMultiVector")
        .def(py::init(&createRCPPyMultiVector1))
        .def(py::init(&createRCPPyMultiVector2))
        .def(py::init(&createRCPPyMultiVectorEmpty))
        .def("getVector", [](RCP_PyMultiVector &m, int i) {
            return m->getVectorNonConst(i);
        })
        .def("getLocalViewHost",[](RCP_PyMultiVector &m){
            return getLocalViewHost(m);
        })
        .def("setLocalViewHost",[](RCP_PyMultiVector &m, py::array_t<ST> input){
            return setLocalViewHost(m, input);
        })
        .def("getMap",[](RCP_PyMultiVector &m){
            return m->getMap();
        })
        .def("getNumVectors",[](RCP_PyMultiVector &m){
            return m->getNumVectors();
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
        .def(py::init<std::string, Teuchos::RCP<PyParallelEnv>>())
        .def(py::init<Teuchos::RCP<Teuchos::ParameterList>, Teuchos::RCP<PyParallelEnv>>())
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
}
