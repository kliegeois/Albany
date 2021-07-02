//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "LandIce_HydrologyWaterThickness.hpp"

namespace LandIce
{

template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
HydrologyWaterThickness<EvalT, Traits, IsStokes, ThermoCoupled>::
HydrologyWaterThickness (const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl)
{
  /*
   *  The (water) thickness equation has the following (strong) form
   *
   *     dh/dt = m/rho_i + (h_r-h)*|u_b|/l_r - c_creep*AhN^3
   *
   *  where h is the water thickness, m the melting rate of the ice,
   *  h_r/l_r typical height/length of bed bumps, u_b the sliding
   *  velocity of the ice, A the Glen's law flow factor, and N is
   *  the effective pressure.
   *  NOTE: if the term m/rho_i is present, there is no proof of well posedness
   *        for the hydrology equations. Therefore, we only turn this term
   *        on upon request.
   *
   *  In the steady case, we can solve for h, leading to the expression
   *
   *     h = ( m/rho_i + |u_b|*h_r/l_r ) / ( c_creep*AN^3 + |u_b|/l_r )
   */

  bool nodal = p.get<bool>("Nodal");
  if (IsStokes) {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, std::logic_error,
                                "Error! For coupling with StokesFO, the Layouts structure must be that of the basal side.\n");

    sideSetName = p.get<std::string>("Side Set Name");
  }

  auto layout = nodal ? dl->node_scalar : dl->qp_scalar;
  numPts = layout->extent(1);

  u_b = PHX::MDField<const IceScalarT>(p.get<std::string> ("Sliding Velocity Variable Name"), layout);
  N   = PHX::MDField<const ScalarT>(p.get<std::string> ("Effective Pressure Variable Name"), layout);
  A   = PHX::MDField<const TempScalarT>(p.get<std::string> ("Ice Softness Variable Name"), dl->cell_scalar2);
  h   = PHX::MDField<ScalarT>(p.get<std::string> ("Water Thickness Variable Name"), layout);

  this->addDependentField(u_b);
  this->addDependentField(N);
  this->addDependentField(A);

  this->addEvaluatedField(h);

  // Setting parameters
  Teuchos::ParameterList& hydro_params   = *p.get<Teuchos::ParameterList*>("LandIce Hydrology Parameters");
  Teuchos::ParameterList& cav_eqn_params = hydro_params.sublist("Cavities Equation");
  Teuchos::ParameterList& phys_params    = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");

  rho_i = phys_params.get<double>("Ice Density");
  h_r = cav_eqn_params.get<double>("Bed Bumps Height");
  l_r = cav_eqn_params.get<double>("Bed Bumps Length");
  c_creep = cav_eqn_params.get<double>("Creep Closure Coefficient",1.0);

  use_melting = cav_eqn_params.get<bool>("Use Melting", false);
  if (use_melting) {
    m = PHX::MDField<const ScalarT>(p.get<std::string> ("Melting Rate Variable Name"), layout);
    this->addDependentField(m);
  }

  /*
   * Scalings, needed to account for different units: ice velocity
   * is in m/yr, the mesh is in km, and hydrology time unit is s.
   *
   * The steady cavity equation has 3 terms (here in strong form, without signs), with the following units:
   *
   *  2) m/rho_i          [m yr^-1]
   *  3) c_creep*A*h*N^3  [m s^-1 ]
   *  4) (h_r-h)*|u|/l_r  [m yr^-1]
   *
   * We decide to uniform all terms to have units [m yr^-1].
   * Where possible, we do this by rescaling some constants. Otherwise,
   * we simply introduce a new scaling factor
   *
   *  1) m/rho_i              (no scaling)
   *  2) scaling_A*A*h*N^3    scaling_A = yr_to_s
   *  3) (h_r-h)*|u|/l_r      (no scaling)
   *
   * where yr_to_s=365.25*24*3600 (the number of seconds in a year)
   */

  double yr_to_s = 365.25*24*3600;
  c_creep *= yr_to_s;

  this->setName("HydrologyWaterThickness"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
void HydrologyWaterThickness<EvalT, Traits, IsStokes, ThermoCoupled>::evaluateFields (typename Traits::EvalData workset)
{
  if (IsStokes) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
void HydrologyWaterThickness<EvalT, Traits, IsStokes, ThermoCoupled>::evaluateFieldsCell (typename Traits::EvalData workset)
{
  ScalarT zero (0.0);
  for (unsigned int cell=0; cell < workset.numCells; ++cell)
  {
    for (unsigned int ipt=0; ipt < numPts; ++ipt)
    {
      h(cell,ipt)  = (use_melting ? m(cell,ipt)/rho_i : zero) + u_b(cell,ipt)*h_r/l_r;
      h(cell,ipt) /= c_creep*A(cell)*std::pow(N(cell,ipt),3) + u_b(cell,ipt)/l_r;
    }
  }
}

template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
void HydrologyWaterThickness<EvalT, Traits, IsStokes, ThermoCoupled>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
    return;

  ScalarT zero (0.0);
  const auto& sideSet = workset.sideSets->at(sideSetName);
  for (unsigned int side=0; side<sideSet.size(); ++side) {

    // Get the local data of side and cell
    const int cell = sideSet[side].elem_LID;

    for (unsigned int ipt=0; ipt < numPts; ++ipt)
    {
      h(cell,ipt)  = (use_melting ? m(side,ipt)/rho_i : zero) + u_b(side,ipt)*h_r/l_r;
      h(cell,ipt) /= c_creep*A(side)*std::pow(N(side,ipt),3) + u_b(side,ipt)/l_r;
    }
  }
}

} // Namespace LandIce
