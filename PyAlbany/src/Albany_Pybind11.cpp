#include "Teuchos_RCP.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_DefaultComm.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <mpi.h>
#include <mpi4py/mpi4py.h>

using Teuchos_Comm_PyAlbany = Teuchos::Comm<int>;
using RCP_Teuchos_Comm_PyAlbany = Teuchos::RCP<const Teuchos_Comm_PyAlbany >;

namespace py = pybind11;

#define PyString_Check(name) PyBytes_Check(name)

struct mpi4py_comm {
  mpi4py_comm() = default;
  mpi4py_comm(MPI_Comm value) : value(value) {}
  operator MPI_Comm () { return value; }

  MPI_Comm value;
};

namespace pybind11 { namespace detail {
  template <> struct type_caster<mpi4py_comm> {
    public:
      PYBIND11_TYPE_CASTER(mpi4py_comm, _("mpi4py_comm"));

      // Python -> C++
      bool load(handle src, bool) {
        if (import_mpi4py() < 0) {
          throw py::error_already_set();
        }

        PyObject *py_src = src.ptr();

        // Check that we have been passed an mpi4py communicator
        if (PyObject_TypeCheck(py_src, &PyMPIComm_Type)) {
          // Convert to regular MPI communicator
          value.value = *PyMPIComm_Get(py_src);
          return !PyErr_Occurred();
        }

        return false;
      }

      // C++ -> Python
      static handle cast(mpi4py_comm src,
                         return_value_policy /* policy */,
                         handle /* parent */)
      {
        // Create an mpi4py handle
        return PyMPIComm_New(src.value);
      }
  };
}} // namespace pybind11::detail

RCP_Teuchos_Comm_PyAlbany
getDefaultComm (std::vector<std::string> stdvec_args) {

    int argc = (int)stdvec_args.size();
    char **argv = new char*[argc+1];
    for (int i = 0; i < argc; ++i) {
        argv[i] = (char*)stdvec_args[i].data();
    }
    argv[argc] = nullptr;

    MPI_Init(&argc, &argv);

    return Teuchos::DefaultComm<int>::getComm();
}

RCP_Teuchos_Comm_PyAlbany
getTeuchosComm (mpi4py_comm comm) {
    return Teuchos::rcp<const Teuchos_Comm_PyAlbany>(new Teuchos::MpiComm< int >
      (Teuchos::opaqueWrapper(comm.value)));
}

void finalize() {
    MPI_Finalize();
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
    m.def("getTeuchosComm", &getTeuchosComm, "A function which multiplies two numbers");
    m.def("finalize", &finalize, "A function which multiplies two numbers");
    //m.def("getDefaultComm", [](PyObject *args) { return getDefaultComm(args); }, py::arg("args"));
}
