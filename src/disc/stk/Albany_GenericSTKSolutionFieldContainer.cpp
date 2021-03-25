//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_GenericSTKSolutionFieldContainer.hpp"
#include "Albany_GenericSTKSolutionFieldContainer_Def.hpp"

namespace Albany {

template class GenericSTKSolutionFieldContainer<DiscType::BlockedMono>;
template class GenericSTKSolutionFieldContainer<DiscType::Interleaved>;
template class GenericSTKSolutionFieldContainer<DiscType::BlockedDisc>;

} // namespace Albany
