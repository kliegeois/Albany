//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_MULTI_STK_FIELD_CONTAINER_HPP
#define ALBANY_MULTI_STK_FIELD_CONTAINER_HPP

#include "Albany_GenericSTKFieldContainer.hpp"
#include "Teuchos_Array.hpp"

namespace Albany {

template <DiscType Interleaved>
class MultiSTKFieldContainer : public GenericSTKFieldContainer<Interleaved>
{
 public:
  MultiSTKFieldContainer(
      const Teuchos::RCP<Teuchos::ParameterList>&               params_,
      const Teuchos::RCP<stk::mesh::MetaData>&                  metaData_,
      const Teuchos::RCP<stk::mesh::BulkData>&                  bulkData_,
      const int                                                 numDim_,
      const Teuchos::RCP<Albany::StateInfoStruct>&              sis,
      const Teuchos::Array<Teuchos::Array<std::string>>&        solution_vector,
      const int                                                 num_params);

  ~MultiSTKFieldContainer() = default;

  void
  fillVector(
      Thyra_Vector&                                field_vector,
      const std::string&                           field_name,
      stk::mesh::Selector&                         field_selection,
      const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
      const NodalDOFManager&                       nodalDofManager);
  void
  saveVector(
      const Thyra_Vector&                          field_vector,
      const std::string&                           field_name,
      stk::mesh::Selector&                         field_selection,
      const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
      const NodalDOFManager&                       nodalDofManager);

  int getNumParams() {return num_params;}

 private:
  void
  fillVectorImpl(
      Thyra_Vector&                                field_vector,
      const std::string&                           field_name,
      stk::mesh::Selector&                         field_selection,
      const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
      const NodalDOFManager&                       nodalDofManager,
      const int                                    offset);
  void
  saveVectorImpl(
      const Thyra_Vector&                          field_vector,
      const std::string&                           field_name,
      stk::mesh::Selector&                         field_selection,
      const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
      const NodalDOFManager&                       nodalDofManager,
      const int                                    offset);

  void
  initializeProcRankField();

  // Containers for residual and solution


  Teuchos::Array<std::string> res_vector_name;
  Teuchos::Array<int>         res_index;
  int num_params;
};

}  // namespace Albany

#endif // ALBANY_MULTI_STK_FIELD_CONTAINER_HPP
