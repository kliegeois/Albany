#ifndef PYALBANY_COMM_H
#define PYALBANY_COMM_H

#include <mpi.h>
#include <mpi4py/mpi4py.h>

using Teuchos_Comm_PyAlbany = Teuchos::Comm<int>;
using RCP_Teuchos_Comm_PyAlbany = Teuchos::RCP<const Teuchos_Comm_PyAlbany >;

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
          throw pybind11::error_already_set();
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

bool
inializeMPI (std::vector<std::string> stdvec_args) {
    int ierr = 0;
    MPI_Initialized(&ierr);
    if (!ierr) {
      int argc = (int)stdvec_args.size();
      char **argv = new char*[argc+1];
      for (int i = 0; i < argc; ++i) {
          argv[i] = (char*)stdvec_args[i].data();
      }
      argv[argc] = nullptr;

      MPI_Init(&argc, &argv);
      return true;
    }

    return false;
}

RCP_Teuchos_Comm_PyAlbany
getDefaultComm () {
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

PyObject * reduceAll(RCP_Teuchos_Comm_PyAlbany comm, Teuchos::EReductionType reductOp, PyObject * sendObj)
{
    return NULL;
}

#endif
