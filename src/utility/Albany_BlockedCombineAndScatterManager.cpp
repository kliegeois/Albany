#include "Albany_BlockedCombineAndScatterManager.hpp"

#include "Albany_ThyraUtils.hpp"

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

    auto pvs_owned = Teuchos::rcp_dynamic_cast<const Thyra_ProductVectorSpace>(owned);
    auto pvs_overlapped = Teuchos::rcp_dynamic_cast<const Thyra_ProductVectorSpace>(overlapped);

    n_blocks = pvs_owned->numBlocks();

    cas_blocks.resize(n_blocks);
    for (int i=0; i<n_blocks; ++i)
        cas_blocks[i] = createCombineAndScatterManager(pvs_owned->getBlock(i), pvs_overlapped->getBlock(i));

}

void BlockedCombineAndScatterManager::
combine (const Thyra_Vector& src,
               Thyra_Vector& dst,
         const CombineMode CM) const
{
    for (int i=0; i<n_blocks; ++i)
        cas_blocks[i]->combine(src, dst, CM);
}

void BlockedCombineAndScatterManager::
combine (const Thyra_MultiVector& src,
               Thyra_MultiVector& dst,
         const CombineMode CM) const
{
    for (int i=0; i<n_blocks; ++i)
        cas_blocks[i]->combine(src, dst, CM);
}

void BlockedCombineAndScatterManager::
combine (const Thyra_LinearOp& src,
               Thyra_LinearOp& dst,
         const CombineMode CM) const
{
    for (int i=0; i<n_blocks; ++i)
        cas_blocks[i]->combine(src, dst, CM);
}

void BlockedCombineAndScatterManager::
combine (const Teuchos::RCP<const Thyra_Vector>& src,
         const Teuchos::RCP<      Thyra_Vector>& dst,
         const CombineMode CM) const
{
    auto src_pv = getConstProductVector(src, false);
    auto dst_pv = getProductVector(dst, false);

    TEUCHOS_TEST_FOR_EXCEPTION (src_pv.is_null(), std::runtime_error, "Source vector is not product based.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (dst_pv.is_null(), std::runtime_error, "Destination vector is not product based.\n");

    for (int i=0; i<n_blocks; ++i)
        cas_blocks[i]->combine(src_pv->getVectorBlock(i), dst_pv->getNonconstVectorBlock(i), CM);
}

void BlockedCombineAndScatterManager::
combine (const Teuchos::RCP<const Thyra_MultiVector>& src,
         const Teuchos::RCP<      Thyra_MultiVector>& dst,
         const CombineMode CM) const
{
    auto src_pv = getConstProductMultiVector(src, false);
    auto dst_pv = getProductMultiVector(dst, false);

    TEUCHOS_TEST_FOR_EXCEPTION (src_pv.is_null(), std::runtime_error, "Source vector is not product based.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (dst_pv.is_null(), std::runtime_error, "Destination vector is not product based.\n");

    for (int i=0; i<n_blocks; ++i)
        cas_blocks[i]->combine(src_pv->getMultiVectorBlock(i), dst_pv->getNonconstMultiVectorBlock(i), CM);
}

void BlockedCombineAndScatterManager::
combine (const Teuchos::RCP<const Thyra_LinearOp>& src,
         const Teuchos::RCP<      Thyra_LinearOp>& dst,
         const CombineMode CM) const
{
    for (int i=0; i<n_blocks; ++i)
        cas_blocks[i]->combine(src, dst, CM);
}

// Scatter methods
void BlockedCombineAndScatterManager::
scatter (const Thyra_Vector& src,
               Thyra_Vector& dst,
         const CombineMode CM) const
{
    for (int i=0; i<n_blocks; ++i)
        cas_blocks[i]->scatter(src, dst, CM);
}

void BlockedCombineAndScatterManager::
scatter (const Thyra_MultiVector& src,
               Thyra_MultiVector& dst,
         const CombineMode CM) const
{
    for (int i=0; i<n_blocks; ++i)
        cas_blocks[i]->scatter(src, dst, CM);
}

void BlockedCombineAndScatterManager::
scatter (const Thyra_LinearOp& src,
               Thyra_LinearOp& dst,
         const CombineMode CM) const
{
    for (int i=0; i<n_blocks; ++i)
        cas_blocks[i]->scatter(src, dst, CM);
}

void BlockedCombineAndScatterManager::
scatter (const Teuchos::RCP<const Thyra_Vector>& src,
         const Teuchos::RCP<      Thyra_Vector>& dst,
         const CombineMode CM) const
{
    auto src_pv = getConstProductVector(src, false);
    auto dst_pv = getProductVector(dst, false);

    TEUCHOS_TEST_FOR_EXCEPTION (src_pv.is_null(), std::runtime_error, "Source vector is not product based.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (dst_pv.is_null(), std::runtime_error, "Destination vector is not product based.\n");

    for (int i=0; i<n_blocks; ++i)
        cas_blocks[i]->scatter(src_pv->getVectorBlock(i), dst_pv->getNonconstVectorBlock(i), CM);
}

void BlockedCombineAndScatterManager::
scatter (const Teuchos::RCP<const Thyra_MultiVector>& src,
         const Teuchos::RCP<      Thyra_MultiVector>& dst,
         const CombineMode CM) const
{
    auto src_pv = getConstProductMultiVector(src, false);
    auto dst_pv = getProductMultiVector(dst, false);

    TEUCHOS_TEST_FOR_EXCEPTION (src_pv.is_null(), std::runtime_error, "Source vector is not product based.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (dst_pv.is_null(), std::runtime_error, "Destination vector is not product based.\n");

    for (int i=0; i<n_blocks; ++i)
        cas_blocks[i]->scatter(src_pv->getMultiVectorBlock(i), dst_pv->getNonconstMultiVectorBlock(i), CM);
}

void BlockedCombineAndScatterManager::
scatter (const Teuchos::RCP<const Thyra_LinearOp>& src,
         const Teuchos::RCP<      Thyra_LinearOp>& dst,
         const CombineMode CM) const
{
    for (int i=0; i<n_blocks; ++i)
        cas_blocks[i]->scatter(src, dst, CM);
}

void BlockedCombineAndScatterManager::
create_ghosted_aura_owners () const {

}

void BlockedCombineAndScatterManager::
create_owned_aura_users () const {

}

} // namespace Albany
