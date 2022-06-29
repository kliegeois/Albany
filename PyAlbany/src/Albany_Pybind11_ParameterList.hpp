#ifndef PYALBANY_PARAMETERLIST_H
#define PYALBANY_PARAMETERLIST_H

#include "Albany_Utils.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_YamlParameterListHelpers.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using PyParameterList = Teuchos::ParameterList;
using RCP_PyParameterList = Teuchos::RCP<PyParameterList>;

template< typename T >
void copyNumPyToTeuchosArray(PyObject * pyArray,
                             Teuchos::Array< T > & tArray);

bool setPythonParameter(Teuchos::ParameterList & plist,
			const std::string      & name,
			pybind11::object             value);

template< typename T > 
pybind11::object copyTeuchosArrayToNumPy(Teuchos::Array< T > & tArray);

pybind11::object getPythonParameter(const Teuchos::ParameterList & plist, const std::string & name);

RCP_PyParameterList createRCPPyParameterList();

void pyalbany_parameterlist(pybind11::module &m);

#endif
