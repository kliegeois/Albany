//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_MULTI_STK_SOLUTION_FIELD_CONTAINER_HPP
#define ALBANY_MULTI_STK_SOLUTION_FIELD_CONTAINER_HPP

#include "Albany_GenericSTKSolutionFieldContainer.hpp"
#include "Albany_MultiSTKFieldContainer.hpp"
#include "Teuchos_Array.hpp"

namespace Albany {

template <DiscType Interleaved>
class MultiSTKSolutionFieldContainer : public GenericSTKSolutionFieldContainer<Interleaved>
{
 public:
  MultiSTKSolutionFieldContainer(
    const Teuchos::RCP<Teuchos::ParameterList>& params_,
    const int numDim_,
    const int neq_,
    const Teuchos::RCP<MultiSTKFieldContainer<Interleaved>>& fieldContainer_,
    const Teuchos::Array<Teuchos::Array<std::string>>& solution_vector,
    const int                                          num_params_);

  ~MultiSTKSolutionFieldContainer() = default;

  void
  fillSolnVector(
      Thyra_Vector&                                soln,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void
  fillVector(
      Thyra_Vector&                                field_vector,
      const std::string&                           field_name,
      stk::mesh::Selector&                         field_selection,
      const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
      const NodalDOFManager&                       nodalDofManager);
  void
  fillSolnMultiVector(
      Thyra_MultiVector&                           soln,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void
  saveVector(
      const Thyra_Vector&                          field_vector,
      const std::string&                           field_name,
      stk::mesh::Selector&                         field_selection,
      const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
      const NodalDOFManager&                       nodalDofManager);
  void
  saveSolnVector(
      const Thyra_Vector&                          soln,
      const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void
  saveSolnVector(
      const Thyra_Vector&                          soln,
      const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
      const Thyra_Vector&                          soln_dot,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void
  saveSolnVector(
      const Thyra_Vector&                          soln,
      const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
      const Thyra_Vector&                          soln_dot,
      const Thyra_Vector&                          soln_dotdot,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void
  saveResVector(
      const Thyra_Vector&                          res,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void
  saveSolnMultiVector(
      const Thyra_MultiVector&                     soln,
      const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);

  void
  transferSolutionToCoords();

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

  // Containers for residual and solution

  Teuchos::Array<Teuchos::Array<std::string>> sol_vector_name;
  Teuchos::Array<Teuchos::Array<int>>         sol_index;

  Teuchos::Array<std::string> res_vector_name;
  Teuchos::Array<int>         res_index;
};

}  // namespace Albany

#endif // ALBANY_MULTI_STK_SOLUTION_FIELD_CONTAINER_HPP