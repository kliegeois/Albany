//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(FerroicModel_hpp)
#define FerroicModel_hpp

#include "FerroicCore.hpp"

namespace FM
{

template<typename EvalT>
class FerroicModel
{
public:

  using ScalarT = typename EvalT::ScalarT;

  // Constructor
  //
  FerroicModel();

  // Virtual Denstructor
  //
  ~FerroicModel()
  {
  }

  // Method to compute the state
  //
  void
  computeState(
      const Intrepid2::Tensor<ScalarT, FM::THREE_D>& x,
      const Intrepid2::Vector<ScalarT, FM::THREE_D>& E,
      const Teuchos::Array<RealType>& oldfractions,
            Intrepid2::Tensor<ScalarT, FM::THREE_D>& X,
            Intrepid2::Vector<ScalarT, FM::THREE_D>& D,
            Teuchos::Array<ScalarT>& newfractions);
     

  // Accessors
  //
  Intrepid2::Tensor<RealType, FM::THREE_D>&          
  getBasis() { return R; }

  Teuchos::Array<RealType>&
  getInitialBinFractions() { return initialBinFractions; }

  Teuchos::Array<Teuchos::RCP<FM::CrystalPhase> >&
  getCrystalPhases() { return crystalPhases; }

  Teuchos::Array<FM::CrystalVariant>&
  getCrystalVariants() { return crystalVariants; }

  std::vector< FM::Transition >&
  getTransitions() { return transitions; }

  Teuchos::Array<RealType>&
  getTransitionBarrier() { return tBarriers; }

  void PostParseInitialize();

private:

  ///
  /// Private to prohibit copying
  ///
  FerroicModel(const FerroicModel&);

  ///
  /// Private to prohibit copying
  ///
  FerroicModel& operator=(const FerroicModel&);

  // parameters
  //
  Intrepid2::Tensor<RealType, FM::THREE_D>        R;
  Teuchos::Array<Teuchos::RCP<FM::CrystalPhase> > crystalPhases;
  Teuchos::Array<FM::CrystalVariant>              crystalVariants;
  Teuchos::Array<FM::Transition>                  transitions;
  Teuchos::Array<RealType>                        tBarriers;
  Teuchos::Array<RealType>                        initialBinFractions;
  Intrepid::FieldContainer<RealType>              aMatrix;

  // Solution options
  //
  IntegrationType       m_integrationType;
  ExplicitMethod        m_explicitMethod;
  Intrepid2::StepType   m_step_type;

  RealType              m_implicit_nonlinear_solver_relative_tolerance_;
  RealType              m_implicit_nonlinear_solver_absolute_tolerance_;
  int                   m_implicit_nonlinear_solver_max_iterations_;
  int                   m_implicit_nonlinear_solver_min_iterations_;

};

}

#include "FerroicModel_Def.hpp"

#endif
