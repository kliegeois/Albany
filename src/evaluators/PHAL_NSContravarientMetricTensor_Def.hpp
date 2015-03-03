//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
NSContravarientMetricTensor<EvalT, Traits>::
NSContravarientMetricTensor(const Teuchos::ParameterList& p) :
  coordVec      (p.get<std::string>                   ("Coordinate Vector Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout") ),
  cubature      (p.get<Teuchos::RCP <Intrepid::Cubature<RealType> > >("Cubature")),
  cellType      (p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type")),
  Gc            (p.get<std::string>                   ("Contravarient Metric Tensor Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") )
{
  this->addDependentField(coordVec);
  this->addEvaluatedField(Gc);

  // Get Dimensions
  Teuchos::RCP<PHX::DataLayout> vector_dl = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dim;
  vector_dl->dimensions(dim);
  int containerSize = dim[0];
  numQPs = dim[1];
  numDims = dim[2];

  // Allocate Temporary FieldContainers
  refPoints.resize(numQPs, numDims);
  refWeights.resize(numQPs);
  jacobian.resize(containerSize, numQPs, numDims, numDims);
  jacobian_inv.resize(containerSize, numQPs, numDims, numDims);

  // Pre-Calculate reference element quantitites
  cubature->getCubature(refPoints, refWeights);

  this->setName("NSContravarientMetricTensor" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSContravarientMetricTensor<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(Gc,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSContravarientMetricTensor<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  /** The allocated size of the Field Containers must currently 
    * match the full workset size of the allocated PHX Fields, 
    * this is the size that is used in the computation. There is
    * wasted effort computing on zeroes for the padding on the
    * final workset. Ideally, these are size numCells.
  //int containerSize = workset.numCells;
    */
  
  Intrepid::CellTools<MeshScalarT>::setJacobian(jacobian, refPoints, coordVec, *cellType);
  Intrepid::CellTools<MeshScalarT>::setJacobianInv(jacobian_inv, jacobian);

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {      
      for (std::size_t i=0; i < numDims; ++i) {        
        for (std::size_t j=0; j < numDims; ++j) {
          Gc(cell,qp,i,j) = 0.0;
          for (std::size_t alpha=0; alpha < numDims; ++alpha) {  
            Gc(cell,qp,i,j) += jacobian_inv(cell,qp,alpha,i)*jacobian_inv(cell,qp,alpha,j); 
          }
        } 
      } 
    }
  }
  
}

//**********************************************************************
}
