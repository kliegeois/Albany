#ifndef PYALBANY_TPETRA_H
#define PYALBANY_TPETRA_H

#include "Albany_TpetraTypes.hpp"

using RCP_PyMap = Teuchos::RCP<Tpetra_Map>;
using RCP_ConstPyMap = Teuchos::RCP<const Tpetra_Map>;
using RCP_PyVector = Teuchos::RCP<Tpetra_Vector>;
using RCP_PyMultiVector = Teuchos::RCP<Tpetra_MultiVector>;

template< bool B, class T = void >
using enable_if_t = typename std::enable_if<B,T>::type;

template<typename T>
Teuchos::ArrayView< T > convert_np_to_ArrayView(pybind11::array_t<T> array) {

    auto np_array = array.template mutable_unchecked<1>();
    int size = array.shape(0);
    Teuchos::ArrayView< T > av(array.mutable_data(0), size);

    return av;
}

// conversion of numpy arrays to kokkos arrays

template<typename ViewType>
void convert_np_to_kokkos_1d(pybind11::array_t<typename ViewType::non_const_value_type> array,  ViewType kokkos_array_device) {

    auto np_array = array.template unchecked<1>();

    auto kokkos_array_host = Kokkos::create_mirror_view(kokkos_array_device);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, array.shape(0)), [&](int i) {
        kokkos_array_host(i) = np_array(i);
    });
    Kokkos::fence();
    Kokkos::deep_copy(kokkos_array_device, kokkos_array_host);
}

template<typename ViewType>
void convert_np_to_kokkos_2d(pybind11::array_t<typename ViewType::non_const_value_type> array,  ViewType kokkos_array_device) {

    auto np_array = array.template unchecked<2>();

    auto kokkos_array_host = Kokkos::create_mirror_view(kokkos_array_device);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, array.shape(0)), [&](int i) {
        for (int j=0; j<array.shape(1); ++j) {
            kokkos_array_host(i,j) = np_array(i,j);
        }
    });
    Kokkos::fence();
    Kokkos::deep_copy(kokkos_array_device, kokkos_array_host);
}

// conversion of kokkos arrays to numpy arrays

template<typename T, typename T2=void>
struct cknp1d {
    pybind11::array_t<typename T::value_type> result;
    cknp1d (T kokkos_array_host) {

        auto dim_out_0 = kokkos_array_host.extent(0);
        result = pybind11::array_t<typename T::value_type>(dim_out_0);
        auto data = result.template mutable_unchecked<1>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0), [&](int i) {
            data(i) = kokkos_array_host(i);
        });
        Kokkos::fence();

    }
    pybind11::array_t<typename T::value_type> convert() { return result; }
};

template<typename T>
struct cknp1d<T, enable_if_t<(T::rank!=1)> > {
    pybind11::array_t<typename T::value_type> result;
    cknp1d (T kokkos_array_host) {
        result = pybind11::array_t<typename T::value_type>(0);
    }
    pybind11::array_t<typename T::value_type> convert() { return result; }
};

template<typename T, typename T2=void>
struct cknp2d {
    pybind11::array_t<typename T::value_type> result;
    cknp2d (T kokkos_array_host) {

        auto dim_out_0 = kokkos_array_host.extent(0);
        auto dim_out_1 = kokkos_array_host.extent(1);

        result = pybind11::array_t<typename T::value_type>(dim_out_0*dim_out_1);
        result.resize({dim_out_0,dim_out_1});
        auto data = result.template mutable_unchecked<T::rank>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0), [&](int i) {
            for (int j=0; j<dim_out_1; ++j) {
                data(i,j) = kokkos_array_host(i,j);
            }
        });
        Kokkos::fence();

    }
    pybind11::array_t<typename T::value_type> convert() { return result; }
};

template<typename T>
struct cknp2d<T, enable_if_t<(T::rank!=2)> > {
    pybind11::array_t<typename T::value_type> result;
    cknp2d (T kokkos_array_host) {
        result = pybind11::array_t<typename T::value_type>(0);
    }
    pybind11::array_t<typename T::value_type> convert() { return result; }
};


template<typename T>
pybind11::array_t<typename T::value_type> convert_kokkos_to_np(T kokkos_array_device) {

    // ensure data is accessible
    auto kokkos_array_host =
        Kokkos::create_mirror_view(kokkos_array_device);
    Kokkos::deep_copy(kokkos_array_host, kokkos_array_device);

    pybind11::array_t<typename T::value_type> result;
    if (T::rank==1) {
        result = cknp1d<decltype(kokkos_array_host)>(kokkos_array_host).convert();
    } else if (T::rank==2) {
        result = cknp2d<decltype(kokkos_array_host)>(kokkos_array_host).convert();
    } else {
        result = pybind11::array_t<typename T::value_type>(0);
    }
    return result;

}

RCP_PyMap createRCPPyMapEmpty() {
    return Teuchos::rcp<Tpetra_Map>(new Tpetra_Map());
}

RCP_PyMap createRCPPyMap(int numGlobalEl, int numMyEl, int indexBase, RCP_Teuchos_Comm_PyAlbany comm ) {
    return Teuchos::rcp<Tpetra_Map>(new Tpetra_Map(numGlobalEl, numMyEl, indexBase, comm));
}

RCP_PyMap createRCPPyMapFromView(int numGlobalEl, pybind11::array_t<int> indexList, int indexBase, RCP_Teuchos_Comm_PyAlbany comm ) {
    Kokkos::View<Tpetra_GO*, Kokkos::DefaultExecutionSpace> indexView("map index view", indexList.shape(0));
    convert_np_to_kokkos_1d(indexList, indexView);
    return Teuchos::rcp<Tpetra_Map>(new Tpetra_Map(numGlobalEl, indexView, indexBase, comm));
}

RCP_PyVector createRCPPyVectorEmpty() {
    return Teuchos::rcp<Tpetra_Vector>(new Tpetra_Vector());
}

RCP_PyVector createRCPPyVector1(RCP_PyMap &map, const bool zeroOut) {
    return Teuchos::rcp<Tpetra_Vector>(new Tpetra_Vector(map, zeroOut));
}

RCP_PyVector createRCPPyVector2(RCP_ConstPyMap &map, const bool zeroOut) {
    return Teuchos::rcp<Tpetra_Vector>(new Tpetra_Vector(map, zeroOut));
}

RCP_PyMultiVector createRCPPyMultiVectorEmpty() {
    return Teuchos::rcp<Tpetra_MultiVector>(new Tpetra_MultiVector());
}

RCP_PyMultiVector createRCPPyMultiVector1(RCP_PyMap &map, const int n_cols, const bool zeroOut) {
    return Teuchos::rcp<Tpetra_MultiVector>(new Tpetra_MultiVector(map, n_cols, zeroOut));
}

RCP_PyMultiVector createRCPPyMultiVector2(RCP_ConstPyMap &map, const int n_cols, const bool zeroOut) {
    return Teuchos::rcp<Tpetra_MultiVector>(new Tpetra_MultiVector(map, n_cols, zeroOut));
}

pybind11::array_t<ST> getLocalViewHost(RCP_PyVector &vector) {
    return convert_kokkos_to_np(vector->getLocalViewDevice(Tpetra::Access::ReadOnly));
}

pybind11::array_t<ST> getLocalViewHost(RCP_PyMultiVector &mvector) {
    return convert_kokkos_to_np(mvector->getLocalViewDevice(Tpetra::Access::ReadOnly));
}

void setLocalViewHost(RCP_PyVector &vector, pybind11::array_t<double> input) {
    auto view = vector->getLocalViewDevice(Tpetra::Access::ReadWrite);
    convert_np_to_kokkos_2d(input, view);
}

void setLocalViewHost(RCP_PyMultiVector &mvector, pybind11::array_t<double> input) {
    auto view = mvector->getLocalViewDevice(Tpetra::Access::ReadWrite);
    convert_np_to_kokkos_2d(input, view);
}

pybind11::tuple getRemoteIndexList(RCP_ConstPyMap map, pybind11::array_t<Tpetra_GO> globalIndexes)
{
    auto globalIndexes_av = convert_np_to_ArrayView(globalIndexes);

    Tpetra::LookupStatus result;
    Teuchos::ArrayView< const Tpetra_GO > globalList(globalIndexes_av);

    pybind11::array_t<int> nodeList_np(globalList.size());
    pybind11::array_t<Tpetra_LO> localList_np(globalList.size());

    Teuchos::ArrayView< int >             nodeList(nodeList_np.mutable_data(0), globalList.size());
    Teuchos::ArrayView< Tpetra_LO >       localList(localList_np.mutable_data(0), globalList.size());

    // Call the method
    result = map->getRemoteIndexList(globalList,
                                      nodeList,
                                      localList);
    
    return pybind11::make_tuple(nodeList_np, localList_np, static_cast< long >(result));
}

#endif
