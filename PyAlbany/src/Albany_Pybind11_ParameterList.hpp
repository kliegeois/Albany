#include "Albany_Utils.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_YamlParameterListHelpers.hpp"

using PyParameterList = Teuchos::ParameterList;
using RCP_PyParameterList = Teuchos::RCP<PyParameterList>;

namespace py = pybind11;

RCP_PyParameterList getParameterList(std::string inputFile, RCP_PyParallelEnv pyParallelEnv)
{
    RCP_PyParameterList params = Teuchos::createParameterList("Albany Parameters");

    const std::string input_extension = Albany::getFileExtension(inputFile);

    if (input_extension == "yaml" || input_extension == "yml")
    {
        Teuchos::updateParametersFromYamlFileAndBroadcast(
            inputFile, params.ptr(), *(pyParallelEnv->comm));
    }
    else
    {
        Teuchos::updateParametersFromXmlFileAndBroadcast(
            inputFile, params.ptr(), *(pyParallelEnv->comm));
    }

    return params;
}

// ****************************************************************** //

bool setPythonParameter(Teuchos::ParameterList & plist,
			const std::string      & name,
			py::object             value)
{
  py::handle h = value;

  // Integer values
  if (PyInt_Check(value.ptr ()))
  {
    plist.set(name, h.cast<int>());
  }

  // Floating point values
  else if (PyFloat_Check(value.ptr ()))
  {
    std::cout << " c " << std::endl;
    plist.set(name, h.cast<double>());
  }
  else
  {
    return false;
  }

  // Successful type conversion
  return true;
}    // setPythonParameter

// **************************************************************** //

py::object getPythonParameter(const Teuchos::ParameterList & plist,
			      const std::string            & name)
{
  // If parameter does not exist, return None
  //if (!plist.isParameter(name)) return Py_BuildValue("");

  // Get the parameter entry.  I now deal with the Teuchos::ParameterEntry
  // objects so that I can query the Teuchos::ParameterList without setting
  // the "used" flag to true.
  const Teuchos::ParameterEntry * entry = plist.getEntryPtr(name);
    /*
  // Boolean parameter values
  if (entry->isType< bool >())
  {
    bool value = Teuchos::any_cast< bool >(entry->getAny(false));
    return PyBool_FromLong((long)value);
  }
  // Integer parameter values
  */
  if (entry->isType< int >())
  {
    int value = Teuchos::any_cast< int >(entry->getAny(false));
    return py::cast(value);
  }
  // Double parameter values
  else if (entry->isType< double >())
  {
    double value = Teuchos::any_cast< double >(entry->getAny(false));
    return py::cast(value);
  }
  /*
  // String parameter values
  else if (entry->isType< std::string >())
  {
    std::string value = Teuchos::any_cast< std::string >(entry->getAny(false));
    return PyString_FromString(value.c_str());
  }
  // Char * parameter values
  else if (entry->isType< char * >())
  {
    char * value = Teuchos::any_cast< char * >(entry->getAny(false));
    return PyString_FromString(value);
  }
*/
  // All  other types are unsupported
  //return NULL;
}    // getPythonParameter

// **************************************************************** //

RCP_PyParameterList createPyParameterList() {
    return Teuchos::rcp<PyParameterList>(new PyParameterList());
}