//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"
#include "Albany_UnitTestSetupHelpers.hpp"
#include "OmegahConnManager.hpp"
#include "Albany_CommUtils.hpp"

#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_UnitTestHelpers.hpp"
#include "Teuchos_LocalTestingHelpers.hpp"

#include "Shards_CellTopology.hpp"

//need to test with field patterns
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Intrepid2_HGRAD_TRI_C1_FEM.hpp"

#include <Omega_h_build.hpp>

#define REQUIRE(cond) \
  TEUCHOS_TEST_FOR_EXCEPTION (!(cond),std::runtime_error, \
      "Condition failed: " << #cond << "\n");

auto createOmegahConnManager(MPI_Comm mpiComm) {
  auto lib = Omega_h::Library(nullptr, nullptr, mpiComm);
  auto mesh = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 2, 2, 0, false);
  return Teuchos::rcp(new Albany::OmegahConnManager(std::move(mesh)));
}

/* copied from tests/unit/disc/UnitTest_BlockedDOFManager.cpp */
template <typename Intrepid2Type>
Teuchos::RCP<const panzer::FieldPattern> buildFieldPattern()
{
  // build a geometric pattern from a single basis
  Teuchos::RCP<Intrepid2::Basis<PHX::exec_space, double, double>> basis = Teuchos::rcp(new Intrepid2Type);
  Teuchos::RCP<const panzer::FieldPattern> pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
  return pattern;
}


TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto teuchosComm = Albany::getDefaultComm();
  auto mpiComm = Albany::getMpiCommFromTeuchosComm(teuchosComm);
  auto conn_mgr = createOmegahConnManager(mpiComm);
  out << "Testing OmegahConnManager constructor\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManagerNoConnClone)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto teuchosComm = Albany::getDefaultComm();
  auto mpiComm = Albany::getMpiCommFromTeuchosComm(teuchosComm);
  auto conn_mgr = createOmegahConnManager(mpiComm);

  auto clone = conn_mgr->noConnectivityClone();
  out << "Testing OmegahConnManager::noConnectivityClone()\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager_getElemsInBlock)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto teuchosComm = Albany::getDefaultComm();
  auto mpiComm = Albany::getMpiCommFromTeuchosComm(teuchosComm);

  auto conn_mgr = createOmegahConnManager(mpiComm);
  auto elmGids = conn_mgr->getElementsInBlock();
  REQUIRE(elmGids.size() == conn_mgr->getOwnedElementCount());
  out << "Testing OmegahConnManager::getElementsInBlock()\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager_getBlockId)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto teuchosComm = Albany::getDefaultComm();
  auto mpiComm = Albany::getMpiCommFromTeuchosComm(teuchosComm);

  auto conn_mgr = createOmegahConnManager(mpiComm);
  std::vector<std::string> blockIds;
  conn_mgr->getElementBlockIds(blockIds);
  REQUIRE(conn_mgr->getBlockId(0) == blockIds[0]);
  const auto nelms = conn_mgr->getOwnedElementCount();
  REQUIRE(blockIds[0] == conn_mgr->getBlockId(nelms-1));
  out << "Testing OmegahConnManager::getBlockId()\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager_getBlockTopologies)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto teuchosComm = Albany::getDefaultComm();
  auto mpiComm = Albany::getMpiCommFromTeuchosComm(teuchosComm);

  auto conn_mgr = createOmegahConnManager(mpiComm);
  std::vector<shards::CellTopology> topoTypes;
  conn_mgr->getElementBlockTopologies(topoTypes);
  shards::CellTopology triTopo(shards::getCellTopologyData< shards::Triangle<3> >());
  REQUIRE(triTopo == topoTypes[0]);
  out << "Testing OmegahConnManager::getElementBlockTopologies()\n";
  success = true;
}

TEUCHOS_UNIT_TEST(OmegahDiscTests, ConnectivityManager_buildConnectivity)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  auto teuchosComm = Albany::getDefaultComm();
  auto mpiComm = Albany::getMpiCommFromTeuchosComm(teuchosComm);

  Teuchos::RCP<const panzer::FieldPattern> patternC1 = buildFieldPattern<Intrepid2::Basis_HGRAD_TRI_C1_FEM<PHX::exec_space, double, double>>();

  auto conn_mgr = createOmegahConnManager(mpiComm);
  conn_mgr->buildConnectivity(*patternC1);
  out << "Testing OmegahConnManager::buildConnectivity()\n";
  success = true;
}
