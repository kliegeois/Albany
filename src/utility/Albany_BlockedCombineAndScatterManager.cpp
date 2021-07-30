#include "Albany_BlockedCombineAndScatterManager.hpp"

#include "Albany_CombineAndScatterManagerTpetra.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#ifdef ALBANY_EPETRA
#include "Albany_CombineAndScatterManagerEpetra.hpp"
#include "Albany_EpetraThyraUtils.hpp"
#endif

namespace Albany
{

BlockedCombineAndScatterManager::
BlockedCombineAndScatterManager(const Teuchos::RCP<const Thyra_VectorSpace>& owned,
                               const Teuchos::RCP<const Thyra_VectorSpace>& overlapped)
 : CombineAndScatterManager(owned,overlapped)
{

}

void BlockedCombineAndScatterManager::
combine (const Thyra_Vector& src,
               Thyra_Vector& dst,
         const CombineMode CM) const
{

}

void BlockedCombineAndScatterManager::
combine (const Thyra_MultiVector& src,
               Thyra_MultiVector& dst,
         const CombineMode CM) const
{

}

void BlockedCombineAndScatterManager::
combine (const Thyra_LinearOp& src,
               Thyra_LinearOp& dst,
         const CombineMode CM) const
{

}

void BlockedCombineAndScatterManager::
combine (const Teuchos::RCP<const Thyra_Vector>& src,
         const Teuchos::RCP<      Thyra_Vector>& dst,
         const CombineMode CM) const
{

}

void BlockedCombineAndScatterManager::
combine (const Teuchos::RCP<const Thyra_MultiVector>& src,
         const Teuchos::RCP<      Thyra_MultiVector>& dst,
         const CombineMode CM) const
{

}

void BlockedCombineAndScatterManager::
combine (const Teuchos::RCP<const Thyra_LinearOp>& src,
         const Teuchos::RCP<      Thyra_LinearOp>& dst,
         const CombineMode CM) const
{

}

// Scatter methods
void BlockedCombineAndScatterManager::
scatter (const Thyra_Vector& src,
               Thyra_Vector& dst,
         const CombineMode CM) const
{

}

void BlockedCombineAndScatterManager::
scatter (const Thyra_MultiVector& src,
               Thyra_MultiVector& dst,
         const CombineMode CM) const
{

}

void BlockedCombineAndScatterManager::
scatter (const Thyra_LinearOp& src,
               Thyra_LinearOp& dst,
         const CombineMode CM) const
{

}

void BlockedCombineAndScatterManager::
scatter (const Teuchos::RCP<const Thyra_Vector>& src,
         const Teuchos::RCP<      Thyra_Vector>& dst,
         const CombineMode CM) const
{

}

void BlockedCombineAndScatterManager::
scatter (const Teuchos::RCP<const Thyra_MultiVector>& src,
         const Teuchos::RCP<      Thyra_MultiVector>& dst,
         const CombineMode CM) const
{

}

void BlockedCombineAndScatterManager::
scatter (const Teuchos::RCP<const Thyra_LinearOp>& src,
         const Teuchos::RCP<      Thyra_LinearOp>& dst,
         const CombineMode CM) const
{

}

void BlockedCombineAndScatterManager::
create_ghosted_aura_owners () const {

}

void BlockedCombineAndScatterManager::
create_owned_aura_users () const {

}

} // namespace Albany
