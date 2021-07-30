#include "Albany_CombineAndScatterManager.hpp"

#include "Albany_BlockedCombineAndScatterManager.hpp"

#include "Albany_CombineAndScatterManagerTpetra.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#ifdef ALBANY_EPETRA
#include "Albany_CombineAndScatterManagerEpetra.hpp"
#include "Albany_EpetraThyraUtils.hpp"
#endif

#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

namespace Albany
{

CombineAndScatterManager::
CombineAndScatterManager (const Teuchos::RCP<const Thyra_VectorSpace>& owned,
                          const Teuchos::RCP<const Thyra_VectorSpace>& overlapped)
 : owned_vs      (owned)
 , overlapped_vs (overlapped)
{
  ghosted_aura_owners_computed = false;
  owned_aura_users_computed = false;
}

void CombineAndScatterManager::create_aura_vss () const {
  auto comm = getComm(overlapped_vs);

  // Count how many processes own each element. This is an all-to-all
  // operation, which we perform with a combine+scatter 
  auto numProcs = Thyra::createMember(owned_vs);
  auto ovNumProcs   = Thyra::createMember(overlapped_vs);
  ovNumProcs->assign(1.0);
  combine(ovNumProcs,numProcs,CombineMode::ADD);
  scatter(numProcs,ovNumProcs,CombineMode::INSERT);
  auto data = getLocalData(ovNumProcs.getConst());

  // If an element is no longer 1.0, then it is in the aura
  auto indexer = createGlobalLocalIndexer(overlapped_vs);
  Teuchos::Array<GO> aura_gids;
  for (int lid=0; lid<data.size(); ++lid) {
    if (data[lid]!=1.0) {
      aura_gids.push_back(indexer->getGlobalElement(lid));
    }
  }

  // Recall the three aura types:
  //  - shared: anything in the overlapped vs that is also
  //    in the overelapped vs of another process
  //  - ghosted: anything in the shared_aura_vs that is not in the owned_vs
  //  - owned: anything in the shared_aura_vs that is also in the owned_vs
  // Obviously, shared=owned+ghosted
  shared_aura_vs  = createVectorSpace(comm,aura_gids);
  ghosted_aura_vs = createVectorSpacesDifference(shared_aura_vs,owned_vs,comm);
  owned_aura_vs   = createVectorSpacesIntersection(shared_aura_vs,owned_vs,comm);
}

// Utility function that returns a concrete manager, depending on the return value
// of Albany::build_type().
Teuchos::RCP<CombineAndScatterManager>
createCombineAndScatterManager (const Teuchos::RCP<const Thyra_VectorSpace>& owned,
                                const Teuchos::RCP<const Thyra_VectorSpace>& overlapped)
{
  Teuchos::RCP<CombineAndScatterManager> manager;

  auto pvs = Teuchos::rcp_dynamic_cast<const Thyra_ProductVectorSpace>(owned);
  if (!pvs.is_null()) {
    manager = Teuchos::rcp( new BlockedCombineAndScatterManager(owned,overlapped) );
  }
  else {
    // Allow failure, since we don't know what the underlying linear algebra is
    auto tvs = getTpetraMap(owned,false);
    if (!tvs.is_null()) {
      // Check that the second vs is also of tpetra type. This time, throw if cast fails.
      tvs = getTpetraMap(overlapped,true);

      manager = Teuchos::rcp( new CombineAndScatterManagerTpetra(owned,overlapped) );
    } else {
  #ifdef ALBANY_EPETRA
      auto evs = getEpetraMap(owned, false);
      if (!evs.is_null()) {
        // Check that the second vs is also of epetra type. This time, throw if cast fails.
        evs = getEpetraMap(overlapped,true);

        manager = Teuchos::rcp( new CombineAndScatterManagerEpetra(owned,overlapped) );
      }
  #endif
    }
  }

  TEUCHOS_TEST_FOR_EXCEPTION (manager.is_null(), std::logic_error, "Error! We were not able to cast the input maps to any of the available concrete implementations (so far, only Epetra and Tpetra).\n");

  return manager;
}

} // namespace Albany
