//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Pybind11_ParameterList.hpp"
#include "Albany_Interface.hpp"

// Implementation based on:
// https://github.com/trilinos/Trilinos/tree/master/packages/PyTrilinos/src/PyTrilinos_Teuchos_Util.cpp

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

template<typename T>
void pyalbany_teuchosarray(py::module &m) {
  std::string pyName = std::string("TArray_")+std::string(typeid(T).name());
  py::class_<Teuchos::Array<T>, Teuchos::RCP<Teuchos::Array<T>>>(m, pyName.c_str(), py::buffer_protocol())
    .def_buffer([](Teuchos::Array<T> &a) -> py::buffer_info {
          return py::buffer_info(
              a.data(),
              sizeof(T),
              py::format_descriptor<T>::format(),
              1,
              { a.size() },
              { sizeof(T) }
          );
      });
}

template <typename T, class... T2>
void define_templated_member_function(py::class_<T2...> &cl)
{
  cl.def("get", [](RCP_PyParameterList &m, const std::string &name) {
      return m->get<T>(name);
  });
  cl.def("set", [](RCP_PyParameterList &m, const std::string &name, T value) {
      return m->set<T>(name, value);
  });  
}

void pyalbany_parameterlist(py::module &m) {
    pyalbany_teuchosarray<int>(m);
    pyalbany_teuchosarray<double>(m);

    py::class_<Teuchos::ParameterList, Teuchos::RCP<Teuchos::ParameterList>> cl(m, "PyParameterList");
        cl.def(py::init<>());
        cl.def("sublist", [](Teuchos::RCP<Teuchos::ParameterList> &m, const std::string &name) {
            if (m->isSublist(name))
                return py::cast(sublist(m, name));
            return py::cast("Invalid sublist name");
        }, py::return_value_policy::reference);
        cl.def("print", [](Teuchos::RCP<Teuchos::ParameterList> &m) {
            m->print();
        });
        cl.def("isParameter", &Teuchos::ParameterList::isParameter);

    define_templated_member_function<bool>(cl);    
    define_templated_member_function<Teuchos::ParameterList>(cl);
    define_templated_member_function<std::string>(cl);
    define_templated_member_function<char*>(cl);
    define_templated_member_function<int>(cl);
    define_templated_member_function<double>(cl);
    define_templated_member_function<Teuchos::Array<int>>(cl);
    define_templated_member_function<Teuchos::Array<double>>(cl);

    m.def("getParameterList", &PyAlbany::getParameterList, "A function which returns an RCP to a parameter list read from file");
}
