#include "Teuchos_RCP.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_DefaultComm.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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
    pyalbany_time(m);

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
