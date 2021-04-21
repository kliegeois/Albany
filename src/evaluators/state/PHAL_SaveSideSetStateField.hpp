//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SAVE_SIDE_SET_STATE_FIELD_HPP
#define PHAL_SAVE_SIDE_SET_STATE_FIELD_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Utilities.hpp"
#include "Albany_Layouts.hpp"

namespace PHAL
{
/** \brief SaveSideSetStatField

*/

template<typename EvalT, typename Traits>
class SaveSideSetStateField : public PHX::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  SaveSideSetStateField (const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData workset);
};

// =========================== SPECIALIZATION ========================= //

template<typename Traits>
class SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>
                    : public PHX::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<PHAL::AlbanyTraits::Residual, Traits>
{
public:

  SaveSideSetStateField (const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields (typename Traits::EvalData d);

private:

  void saveElemState (typename Traits::EvalData d);
  void saveNodeState (typename Traits::EvalData d);

  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;

  Teuchos::RCP<PHX::FieldTag> savestate_operation;
  PHX::MDField<const ScalarT>       field;

  std::string sideSetName;
  std::string fieldName;
  std::string stateName;

  bool nodalState;

  Kokkos::View<int**, PHX::Device> sideNodes;

  Albany::LocalSideSetInfo sideSet;

  MDFieldMemoizer<Traits> memoizer;
};

} // Namespace PHAL

#endif // PHAL_SAVE_SIDE_SET_STATE_FIELD_HPP
