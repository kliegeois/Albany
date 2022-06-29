#include "Albany_Pybind11_ParameterList.hpp"
#include "Albany_Interface.hpp"
#include "Albany_Pybind11_Numpy.hpp"

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

namespace py = pybind11;

template< typename T >
void copyNumPyToTeuchosArray(PyObject * pyArray,
                             Teuchos::Array< T > & tArray)
{
  typedef typename Teuchos::Array< T >::size_type size_type;
  size_type length = PyArray_DIM((PyArrayObject*) pyArray, 0);
  tArray.resize(length);
  T * data = (T*) PyArray_DATA((PyArrayObject*) pyArray);
  for (typename Teuchos::Array< T >::iterator it = tArray.begin();
       it != tArray.end(); ++it)
    *it = *(data++);
}

// ****************************************************************** //

bool setPythonParameter(Teuchos::ParameterList & plist,
			const std::string      & name,
			py::object             value)
{
  py::handle h = value;

  // Boolean values
  if (PyBool_Check(value.ptr ()))
  {
    if (value == Py_True) plist.set(name,true );
    else                  plist.set(name,false);
  }

  // Integer values
  else if (PyInt_Check(value.ptr ()))
  {
    plist.set(name, h.cast<int>());
  }

  // Floating point values
  else if (PyFloat_Check(value.ptr ()))
  {
    plist.set(name, h.cast<double>());
  }

  // Unicode values
  else if (PyUnicode_Check(value.ptr ()))
  {
    PyObject * pyBytes = PyUnicode_AsASCIIString(value.ptr ());
    if (!pyBytes) return false;
    plist.set(name, std::string(PyBytes_AsString(pyBytes)));
    Py_DECREF(pyBytes);
  }

  // String values
  else if (PyString_Check(value.ptr ()))
  {
    plist.set(name, h.cast<std::string>());
  }

/*
  // Sublist values
  else if (PyObject_TypeCheck(value.ptr(), PyParameterList))
  {
    plist.set(name, *(h.cast<RCP_PyParameterList>()));
  }
*/

  // None object not allowed: this is a python type not usable by
  // Trilinos solver packages, so we reserve it for the
  // getPythonParameter() function to indicate that the requested
  // parameter does not exist in the given Teuchos::ParameterList.
  // For logic reasons, this check must come before the check for
  // Teuchos::ParameterList
  else if (value.ptr () == Py_None)
  {
    return false;
  }

  // NumPy arrays and non-dictionary Python sequences
  else if (PyArray_Check(value.ptr ()) || PySequence_Check(value.ptr ()))
  {
    PyObject * pyArray =
      PyArray_CheckFromAny(value.ptr (),
                           NULL,
                           1,
                           1,
                           NPY_ARRAY_DEFAULT | NPY_ARRAY_NOTSWAPPED,
                           NULL);
    if (!pyArray) return false;
    // if (PyArray_TYPE((PyArrayObject*) pyArray) == NPY_BOOL)
    // {
    //   Teuchos::Array< bool > tArray;
    //   copyNumPyToTeuchosArray(pyArray, tArray);
    //   plist.set(name, tArray);
    // }
    // else if (PyArray_TYPE((PyArrayObject*) pyArray) == NPY_INT)
    if (PyArray_TYPE((PyArrayObject*) pyArray) == NPY_INT)
    {
      Teuchos::Array< int > tArray;
      copyNumPyToTeuchosArray(pyArray, tArray);
      plist.set(name, tArray);
    }
    else if (PyArray_TYPE((PyArrayObject*) pyArray) == NPY_LONG)
    {
      Teuchos::Array< long > tArray;
      copyNumPyToTeuchosArray(pyArray, tArray);
      plist.set(name, tArray);
    }
    else if (PyArray_TYPE((PyArrayObject*) pyArray) == NPY_FLOAT)
    {
      Teuchos::Array< float > tArray;
      copyNumPyToTeuchosArray(pyArray, tArray);
      plist.set(name, tArray);
    }
    else if (PyArray_TYPE((PyArrayObject*) pyArray) == NPY_DOUBLE)
    {
      Teuchos::Array< double > tArray;
      copyNumPyToTeuchosArray(pyArray, tArray);
      plist.set(name, tArray);
    }
    else if ((PyArray_TYPE((PyArrayObject*) pyArray) == NPY_STRING) ||
             (PyArray_TYPE((PyArrayObject*) pyArray) == NPY_UNICODE))
    {
      Teuchos::Array< std::string > tArray;
      copyNumPyToTeuchosArray(pyArray, tArray);
      plist.set(name, tArray);
    }
    else
    {
      // Unsupported data type
      if (pyArray != value.ptr ()) Py_DECREF(pyArray);
      return false;
    }
  }

  // All other value types are unsupported
  else
  {
    return false;
  }

  // Successful type conversion
  return true;
}    // setPythonParameter

// **************************************************************** //

template< typename T >
py::object copyTeuchosArrayToNumPy(Teuchos::Array< T > & tArray)
{
  int typecode = NumPy_TypeCode< T >();
  npy_intp dims[] = { tArray.size() };
  PyObject *pyArray = PyArray_SimpleNew(1, dims, typecode);
  T * data = (T*) PyArray_DATA((PyArrayObject*) pyArray);
  for (typename Teuchos::Array< T >::iterator it = tArray.begin();
       it != tArray.end(); ++it)
    *(data++) = *it;
  return py::object(py::handle(pyArray), py::object::stolen_t{});
}

template<>
py::object copyTeuchosArrayToNumPy(Teuchos::Array< std::string > & tArray)
{
  int typecode = NumPy_TypeCode< std::string >();
  npy_intp dims[] = { tArray.size() };
  int strlen = 1;
  for (typename Teuchos::Array< std::string >::iterator it = tArray.begin();
       it != tArray.end(); ++it)
  {
    int itlen = it->size();
    if (itlen > strlen) strlen = itlen;
  }
  py::object pyArray;
  pyArray.ptr() =
    PyArray_New(&PyArray_Type, 1, dims, typecode, NULL, NULL, strlen, 0, NULL);
  char* data = (char*) PyArray_DATA((PyArrayObject*) pyArray.ptr());
  for (typename Teuchos::Array< std::string >::iterator it = tArray.begin();
       it != tArray.end(); ++it)
  {
    strncpy(data, it->c_str(), strlen);
    data += strlen;
  }
  return pyArray;
}

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
  // Boolean parameter values
  if (entry->isType< bool >())
  {
    bool value = Teuchos::any_cast< bool >(entry->getAny(false));
    return py::cast(value);
  }
  // Integer parameter values
  else if (entry->isType< int >())
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
  // String parameter values
  else if (entry->isType< std::string >())
  {
    std::string value = Teuchos::any_cast< std::string >(entry->getAny(false));
    return py::cast(value.c_str());
  }
  // Char * parameter values
  else if (entry->isType< char * >())
  {
    char * value = Teuchos::any_cast< char * >(entry->getAny(false));
    return py::cast(value);
  }

  else if (entry->isArray())
  {
    // try
    // {
    //   Teuchos::Array< bool > tArray =
    //     Teuchos::any_cast< Teuchos::Array< bool > >(entry->getAny(false));
    //   return copyTeuchosArrayToNumPy(tArray);
    // }
    // catch(Teuchos::bad_any_cast &e)
    // {
      try
      {
        Teuchos::Array< int > tArray =
          Teuchos::any_cast< Teuchos::Array< int > >(entry->getAny(false));
        return copyTeuchosArrayToNumPy(tArray);
      }
      catch(Teuchos::bad_any_cast &e)
      {
        try
        {
          Teuchos::Array< long > tArray =
            Teuchos::any_cast< Teuchos::Array< long > >(entry->getAny(false));
          return copyTeuchosArrayToNumPy(tArray);
        }
        catch(Teuchos::bad_any_cast &e)
        {
          try
          {
            Teuchos::Array< float > tArray =
              Teuchos::any_cast< Teuchos::Array< float > >(entry->getAny(false));
            return copyTeuchosArrayToNumPy(tArray);
          }
          catch(Teuchos::bad_any_cast &e)
          {
            try
            {
              Teuchos::Array< double > tArray =
                Teuchos::any_cast< Teuchos::Array< double > >(entry->getAny(false));
              return copyTeuchosArrayToNumPy(tArray);
            }
            catch(Teuchos::bad_any_cast &e)
            {
              try
              {
                Teuchos::Array< std::string > tArray =
                  Teuchos::any_cast< Teuchos::Array< std::string > >(entry->getAny(false));
                return copyTeuchosArrayToNumPy(tArray);
              }
              catch(Teuchos::bad_any_cast &e)
              {
                // Teuchos::Arrays of type other than int or double are
                // currently unsupported
                //return NULL;
              }
            }
          }
        }
      }
    // }
  }

  // All  other types are unsupported
  //return NULL;
}    // getPythonParameter

// **************************************************************** //

RCP_PyParameterList createRCPPyParameterList() {
    return Teuchos::rcp<PyParameterList>(new PyParameterList());
}

void pyalbany_parameterlist(py::module &m) {
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
    m.def("getParameterList", &PyAlbany::getParameterList, "A function which multiplies two numbers");
}
