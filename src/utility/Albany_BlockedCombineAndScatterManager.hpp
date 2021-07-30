#ifndef ALBANY_BLOCKED_COMBINE_AND_SCATTER_MANAGER_HPP
#define ALBANY_BLOCKED_COMBINE_AND_SCATTER_MANAGER_HPP

#include "Albany_CombineAndScatterManager.hpp"

namespace Albany
{

class BlockedCombineAndScatterManager : public CombineAndScatterManager
{
public:
  BlockedCombineAndScatterManager(const Teuchos::RCP<const Thyra_VectorSpace>& owned,
                                 const Teuchos::RCP<const Thyra_VectorSpace>& overlapped);

  // Combine methods
  void combine (const Thyra_Vector& src,
                      Thyra_Vector& dst,
                const CombineMode CM) const override;
  void combine (const Thyra_MultiVector& src,
                      Thyra_MultiVector& dst,
                const CombineMode CM) const override;
  void combine (const Thyra_LinearOp& src,
                      Thyra_LinearOp& dst,
                const CombineMode CM) const override;

  void combine (const Teuchos::RCP<const Thyra_Vector>& src,
                const Teuchos::RCP<      Thyra_Vector>& dst,
                const CombineMode CM) const override;
  void combine (const Teuchos::RCP<const Thyra_MultiVector>& src,
                const Teuchos::RCP<      Thyra_MultiVector>& dst,
                const CombineMode CM) const override;
  void combine (const Teuchos::RCP<const Thyra_LinearOp>& src,
                const Teuchos::RCP<      Thyra_LinearOp>& dst,
                const CombineMode CM) const override;


  // Scatter methods
  void scatter (const Thyra_Vector& src,
                      Thyra_Vector& dst,
                const CombineMode CM) const override;
  void scatter (const Thyra_MultiVector& src,
                      Thyra_MultiVector& dst,
                const CombineMode CM) const override;
  void scatter (const Thyra_LinearOp& src,
                      Thyra_LinearOp& dst,
                const CombineMode CM) const override;

  void scatter (const Teuchos::RCP<const Thyra_Vector>& src,
                const Teuchos::RCP<      Thyra_Vector>& dst,
                const CombineMode CM) const override;
  void scatter (const Teuchos::RCP<const Thyra_MultiVector>& src,
                const Teuchos::RCP<      Thyra_MultiVector>& dst,
                const CombineMode CM) const override;
  void scatter (const Teuchos::RCP<const Thyra_LinearOp>& src,
                const Teuchos::RCP<      Thyra_LinearOp>& dst,
                const CombineMode CM) const override;

protected:
  void create_ghosted_aura_owners () const override;
  void create_owned_aura_users () const override;

  int n_blocks;
  CombineAndScatterManager * block_cas;
};

} // namespace Albany

#endif // ALBANY_BLOCKED_COMBINE_AND_SCATTER_MANAGER_HPP
