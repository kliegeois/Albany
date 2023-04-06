//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "OmegahConnManager.hpp"
#include "Albany_config.h"

#include "Panzer_FieldPattern.hpp"

#include <stk_util/parallel/ParallelReduce.hpp>
#include <stk_mesh/base/GetEntities.hpp>

namespace Albany {

OmegahConnManager::
OmegahConnManager(Omega_h::Mesh& in_mesh) : mesh(in_mesh)
{
  //FIXME does albany support processes without elements?
  TEUCHOS_TEST_FOR_EXCEPTION (!mesh.nelems(), std::runtime_error,
      "Error! Input mesh has no elements!\n");

  // FIXME are we assuming the omegah conn manager will be recreated after each adapt?
  initLocalElmIds();
}

std::vector<GO>
OmegahConnManager::getElementsInBlock (const std::string& /* blockId */) const //FIXME
{
  return std::vector<GO>();
}


void
OmegahConnManager::buildConnectivity(const panzer::FieldPattern &) //FIXME
{
  //FIXME what is a 'FieldPattern'?
}

Teuchos::RCP<panzer::ConnManager>
OmegahConnManager::noConnectivityClone() const
{
  //- for stk this function copies the object without connectivity information
  //- for omegah there is little to no distinction as the mesh object contains the
  //  connectivity
  return Teuchos::RCP(new OmegahConnManager(mesh));
}

const std::vector<LO>&
OmegahConnManager::getAssociatedNeighbors(const LO& /* el */) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
      "Error! Albany does not use elements halos in the mesh, so the method\n"
      "       'OmegahConnManager::getAssociatedNeighbors' should not have been called.\n");

  static std::vector<LO> ret;
  return ret;
}

bool OmegahConnManager::contains (const std::string& sub_part_name) const //FIXME
{
  return false;
}

// Return true if the $subcell_pos-th subcell of dimension $subcell_dim in
// local element $ielem belongs to sub part $sub_part_name
bool OmegahConnManager::belongs (const std::string& sub_part_name, //FIXME
         const LO ielem, const int subcell_dim, const int subcell_pos) const
{
  return false;
}

// Queries the dimension of a part
int OmegahConnManager::part_dim (const std::string&) const
{
  return mesh.dim();
}

} // namespace Albany
