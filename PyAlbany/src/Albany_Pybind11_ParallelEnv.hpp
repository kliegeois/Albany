#ifndef PYALBANY_PARALLELENV_H
#define PYALBANY_PARALLELENV_H

#include "Kokkos_Core.hpp"

class PyParallelEnv
{
public:
    RCP_Teuchos_Comm_PyAlbany comm;
    const int num_threads, num_numa, device_id;

    PyParallelEnv(RCP_Teuchos_Comm_PyAlbany _comm, int _num_threads = -1, int _num_numa = -1, int _device_id = -1) : comm(_comm), num_threads(_num_threads), num_numa(_num_numa), device_id(_device_id)
    {
        Kokkos::InitArguments args;
        args.num_threads = this->num_threads;
        args.num_numa = this->num_numa;
        args.device_id = this->device_id;

        Kokkos::initialize(args);
    }
    ~PyParallelEnv()
    {
        Kokkos::finalize_all();
        if (comm->getRank() == 0)
            std::cout << "~PyParallelEnv()\n";
    }
};

using RCP_PyParallelEnv = Teuchos::RCP<PyParallelEnv>;

RCP_PyParallelEnv createPyParallelEnv(RCP_Teuchos_Comm_PyAlbany _comm, int _num_threads = -1, int _num_numa = -1, int _device_id = -1) {
    return Teuchos::rcp<PyParallelEnv>(new PyParallelEnv(_comm, _num_threads, _num_numa, _device_id));
}

RCP_PyParallelEnv createDefaultKokkosPyParallelEnv(RCP_Teuchos_Comm_PyAlbany _comm) {
    return Teuchos::rcp<PyParallelEnv>(new PyParallelEnv(_comm, -1, -1, -1));
}

#endif
