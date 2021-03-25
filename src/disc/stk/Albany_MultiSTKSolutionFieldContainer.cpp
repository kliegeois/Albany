//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_MultiSTKSolutionFieldContainer.hpp"
#include "Albany_MultiSTKSolutionFieldContainer_Def.hpp"

namespace Albany {

template class MultiSTKSolutionFieldContainer<DiscType::BlockedMono>;
template class MultiSTKSolutionFieldContainer<DiscType::Interleaved>;
template class MultiSTKSolutionFieldContainer<DiscType::BlockedDisc>;

}  // namespace Albany
