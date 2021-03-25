//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_GenericSTKSolutionFieldContainer.hpp"
#include "Albany_STKNodeFieldContainer.hpp"

#include "Albany_Utils.hpp"
#include "Albany_StateInfoStruct.hpp"

// Start of STK stuff
#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>
#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

namespace Albany {

template<DiscType Interleaved>
GenericSTKSolutionFieldContainer<Interleaved>::GenericSTKSolutionFieldContainer(
  const Teuchos::RCP<Teuchos::ParameterList>& params_,
  const int neq_,
  const Teuchos::RCP<GenericSTKFieldContainer<Interleaved>>& fieldContainer_)
  : params(params_),
    neq(neq_) {
      this->stkFieldContainer = fieldContainer_;
}

} // namespace Albany
