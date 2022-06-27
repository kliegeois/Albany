#include "Albany_TpetraTypes.hpp"

using RCP_PyMap = Teuchos::RCP<Tpetra_Map>;
using RCP_PyVector = Teuchos::RCP<Tpetra_Vector>;
using RCP_PyMultiVector = Teuchos::RCP<Tpetra_MultiVector>;

RCP_PyMap createRCPPyMapEmpty() {
    return Teuchos::rcp<Tpetra_Map>(new Tpetra_Map());
}

RCP_PyMap createRCPPyMap(int numGlobalEl, int numMyEl, int indexBase, RCP_Teuchos_Comm_PyAlbany comm ) {
    return Teuchos::rcp<Tpetra_Map>(new Tpetra_Map(numGlobalEl, numMyEl, indexBase, comm));
}

RCP_PyVector createRCPPyVectorEmpty() {
    return Teuchos::rcp<Tpetra_Vector>(new Tpetra_Vector());
}

RCP_PyVector createRCPPyVector(RCP_PyMap &map, const bool zeroOut) {
    return Teuchos::rcp<Tpetra_Vector>(new Tpetra_Vector(map, zeroOut));
}

RCP_PyMultiVector createRCPPyMultiVectorEmpty() {
    return Teuchos::rcp<Tpetra_MultiVector>(new Tpetra_MultiVector());
}

RCP_PyMultiVector createRCPPyMultiVector(RCP_PyMap &map, const int n_cols, const bool zeroOut) {
    return Teuchos::rcp<Tpetra_MultiVector>(new Tpetra_MultiVector(map, n_cols, zeroOut));
}