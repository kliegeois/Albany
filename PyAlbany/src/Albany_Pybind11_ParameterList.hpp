#ifndef PYALBANY_PARAMETERLIST_H
#define PYALBANY_PARAMETERLIST_H

#include "Albany_Utils.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_YamlParameterListHelpers.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using PyParameterList = Teuchos::ParameterList;
using RCP_PyParameterList = Teuchos::RCP<PyParameterList>;

RCP_PyParameterList createRCPPyParameterList();

void pyalbany_parameterlist(pybind11::module &m);

#endif
