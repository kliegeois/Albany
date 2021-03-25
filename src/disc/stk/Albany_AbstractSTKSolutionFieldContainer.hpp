//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ABSTRACT_STK_SOLUTION_FIELD_CONTAINER_HPP
#define ALBANY_ABSTRACT_STK_SOLUTION_FIELD_CONTAINER_HPP

#include "Albany_AbstractSTKFieldContainer.hpp"

namespace Albany {

/*!
 * \brief Abstract interface for an STK solution field container
 *
 */
class AbstractSTKSolutionFieldContainer : public AbstractFieldContainer
{
 public:
  // Tensor per Node/Cell  - (Node, Dim, Dim) or (Cell,Dim,Dim)
  typedef stk::mesh::Field<double, stk::mesh::Cartesian, stk::mesh::Cartesian>
      TensorFieldType;
  // Vector per Node/Cell  - (Node, Dim) or (Cell,Dim)
  typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType;
  // Scalar per Node/Cell  - (Node) or (Cell)
  typedef stk::mesh::Field<double> ScalarFieldType;
  // One int scalar per Node/Cell  - (Node) or (Cell)
  typedef stk::mesh::Field<int> IntScalarFieldType;
  // int vector per Node/Cell  - (Node,Dim/VecDim) or (Cell,Dim/VecDim)
  typedef stk::mesh::Field<int, stk::mesh::Cartesian> IntVectorFieldType;

  typedef stk::mesh::Cartesian QPTag;  // need to invent shards::ArrayDimTag
  // Tensor per QP   - (Cell, QP, Dim, Dim)
  typedef stk::mesh::
      Field<double, QPTag, stk::mesh::Cartesian, stk::mesh::Cartesian>
          QPTensorFieldType;
  // Vector per QP   - (Cell, QP, Dim)
  typedef stk::mesh::Field<double, QPTag, stk::mesh::Cartesian>
      QPVectorFieldType;
  // One scalar per QP   - (Cell, QP)
  typedef stk::mesh::Field<double, QPTag> QPScalarFieldType;

  typedef std::vector<const std::string*> ScalarValueState;
  typedef std::vector<QPScalarFieldType*> QPScalarState;
  typedef std::vector<QPVectorFieldType*> QPVectorState;
  typedef std::vector<QPTensorFieldType*> QPTensorState;

  typedef std::vector<ScalarFieldType*> ScalarState;
  typedef std::vector<VectorFieldType*> VectorState;
  typedef std::vector<TensorFieldType*> TensorState;

  typedef std::map<std::string, double>              MeshScalarState;
  typedef std::map<std::string, std::vector<double>> MeshVectorState;

  typedef std::map<std::string, int>              MeshScalarIntegerState;
  typedef std::map<std::string, GO>               MeshScalarInteger64State;
  typedef std::map<std::string, std::vector<int>> MeshVectorIntegerState;


  AbstractSTKSolutionFieldContainer() : proc_rank_field(nullptr){};


  //! Destructor
  virtual ~AbstractSTKSolutionFieldContainer(){};

  // Coordinates field ALWAYS in 3D
  const VectorFieldType*
  getCoordinatesField3d() const
  {
    return stkFieldContainer->getCoordinatesField3d();
  }
  VectorFieldType*
  getCoordinatesField3d()
  {
    return stkFieldContainer->getCoordinatesField3d();
  }

  const VectorFieldType*
  getCoordinatesField() const
  {
    return stkFieldContainer->getCoordinatesField();
  }
  VectorFieldType*
  getCoordinatesField()
  {
    return stkFieldContainer->getCoordinatesField();
  }

  IntScalarFieldType*
  getProcRankField()
  {
    return stkFieldContainer->getProcRankField();
  }

  ScalarValueState&
  getScalarValueStates()
  {
    return stkFieldContainer->getScalarValueStates();
  }
  MeshScalarState&
  getMeshScalarStates()
  {
    return stkFieldContainer->getMeshScalarStates();
  }
  MeshVectorState&
  getMeshVectorStates()
  {
    return stkFieldContainer->getMeshVectorStates();
  }
  MeshScalarIntegerState&
  getMeshScalarIntegerStates()
  {
    return stkFieldContainer->getMeshScalarIntegerStates();
  }
  MeshScalarInteger64State&
  getMeshScalarInteger64States()
  {
    return stkFieldContainer->getMeshScalarInteger64States();
  }
  MeshVectorIntegerState&
  getMeshVectorIntegerStates()
  {
    return stkFieldContainer->getMeshVectorIntegerStates();
  }
  ScalarState&
  getCellScalarStates()
  {
    return stkFieldContainer->getCellScalarStates();
  }
  VectorState&
  getCellVectorStates()
  {
    return stkFieldContainer->getCellVectorStates();
  }
  TensorState&
  getCellTensorStates()
  {
    return stkFieldContainer->getCellTensorStates();
  }
  QPScalarState&
  getQPScalarStates()
  {
    return stkFieldContainer->getQPScalarStates();
  }
  QPVectorState&
  getQPVectorStates()
  {
    return stkFieldContainer->getQPVectorStates();
  }
  QPTensorState&
  getQPTensorStates()
  {
    return stkFieldContainer->getQPTensorStates();
  }
  const StateInfoStruct&
  getNodalSIS() const
  {
    return stkFieldContainer->getNodalSIS();
  }
  const StateInfoStruct&
  getNodalParameterSIS() const
  {
    return stkFieldContainer->getNodalParameterSIS();
  }

  std::map<std::string, double>&
  getTime()
  {
    return stkFieldContainer->getTime();
  }

  virtual void
  fillSolnVector(
      Thyra_Vector&                                soln,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs) = 0;

  virtual void
  fillVector(
      Thyra_Vector&                                field_vector,
      const std::string&                           field_name,
      stk::mesh::Selector&                         field_selection,
      const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
      const NodalDOFManager&                       nodalDofManager) = 0;

  virtual void
  fillSolnMultiVector(
      Thyra_MultiVector&                           soln,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs) = 0;

  virtual void
  saveVector(
      const Thyra_Vector&                          field_vector,
      const std::string&                           field_name,
      stk::mesh::Selector&                         field_selection,
      const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
      const NodalDOFManager&                       nodalDofManager) = 0;

  virtual void
  saveSolnVector(
      const Thyra_Vector&                          soln,
      const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs) = 0;
  virtual void
  saveSolnVector(
      const Thyra_Vector&                          soln,
      const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
      const Thyra_Vector&                          soln_dot,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs) = 0;
  virtual void
  saveSolnVector(
      const Thyra_Vector&                          soln,
      const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
      const Thyra_Vector&                          soln_dot,
      const Thyra_Vector&                          soln_dotdot,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs) = 0;
  virtual void
  saveResVector(
      const Thyra_Vector&                          res,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs) = 0;
  virtual void
  saveSolnMultiVector(
      const Thyra_MultiVector&                     soln,
      const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs) = 0;

  virtual void
  transferSolutionToCoords() = 0;


 protected:
  // Note: for 3d meshes, coordinates_field3d==coordinates_field (they point to
  // the same field).
  //       Otherwise, coordinates_field3d stores coordinates in 3d (useful for
  //       non-flat 2d meshes)
  VectorFieldType*    coordinates_field3d;
  VectorFieldType*    coordinates_field;
  IntScalarFieldType* proc_rank_field;

  ScalarValueState          scalarValue_states;
  MeshScalarState           mesh_scalar_states;
  MeshVectorState           mesh_vector_states;
  MeshScalarIntegerState    mesh_scalar_integer_states;
  MeshScalarInteger64State  mesh_scalar_integer_64_states;
  MeshVectorIntegerState    mesh_vector_integer_states;
  ScalarState               cell_scalar_states;
  VectorState               cell_vector_states;
  TensorState               cell_tensor_states;
  QPScalarState             qpscalar_states;
  QPVectorState             qpvector_states;
  QPTensorState             qptensor_states;

  StateInfoStruct nodal_sis;
  StateInfoStruct nodal_parameter_sis;

  std::map<std::string, double> time;

  Teuchos::RCP<AbstractSTKFieldContainer> stkFieldContainer;
};

}  // namespace Albany

#endif  // ALBANY_ABSTRACT_STK_SOLUTION_FIELD_CONTAINER_HPP
