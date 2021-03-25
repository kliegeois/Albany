//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_GENERIC_STK_SOLUTION_FIELD_CONTAINER_HPP
#define ALBANY_GENERIC_STK_SOLUTION_FIELD_CONTAINER_HPP

#include "Albany_AbstractSTKSolutionFieldContainer.hpp"
#include "Albany_GenericSTKFieldContainer.hpp"

#include "Teuchos_ParameterList.hpp"

// Forward declaration is enough
namespace stk {
namespace mesh {
class BulkData;
class MetaData;
} // namespace stk
} // namespace mesh

namespace Albany {

template<DiscType Interleaved>
class GenericSTKSolutionFieldContainer : public AbstractSTKSolutionFieldContainer
{
public:

  GenericSTKSolutionFieldContainer(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                           const int neq_,
                           const Teuchos::RCP<GenericSTKFieldContainer<Interleaved>>& fieldContainer_);

  virtual ~GenericSTKSolutionFieldContainer() = default;

protected:

  Teuchos::RCP<Teuchos::ParameterList> params;
  int neq;
};

} // namespace Albany

#endif // ALBANY_GENERIC_STK_SOLUTION_FIELD_CONTAINER_HPP
