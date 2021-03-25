//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_OrdinarySTKSolutionFieldContainer.hpp"
#include "Albany_OrdinarySTKSolutionFieldContainer_Def.hpp"

namespace Albany {

template class OrdinarySTKSolutionFieldContainer<DiscType::BlockedMono>;
template class OrdinarySTKSolutionFieldContainer<DiscType::Interleaved>;
template class OrdinarySTKSolutionFieldContainer<DiscType::BlockedDisc>;

}  // namespace Albany
