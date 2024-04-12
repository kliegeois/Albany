//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <stdexcept>

#include "Albany_Application.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "PHAL_Utilities.hpp"

namespace {

template <typename EvalT> std::string getSFadSizeName();
template <> std::string getSFadSizeName<PHAL::AlbanyTraits::Jacobian>() {return "ALBANY_SFAD_SIZE";}
template <> std::string getSFadSizeName<PHAL::AlbanyTraits::Tangent>() {return "ALBANY_TAN_SFAD_SIZE";}
template <> std::string getSFadSizeName<PHAL::AlbanyTraits::DistParamDeriv>() {return "ALBANY_TAN_SFAD_SIZE";}
template <> std::string getSFadSizeName<PHAL::AlbanyTraits::HessianVec>() {return "ALBANY_HES_VEC_SFAD_SIZE";}

template <typename EvalT>
void checkDerivativeDimensions(const int dDims)
{
  // Check derivative dimensions against fad size
  using FadT = typename EvalT::EvaluationType::ScalarT;
  if (FadT::StorageType::is_statically_sized) {
    const int static_size = FadT::StorageType::static_size;
    if (static_size != dDims) {
      const auto sfadSizeName = getSFadSizeName<EvalT>();
      std::stringstream ss1, ss2;
      ss1 << "Derivative dimension for " << PHX::print<EvalT>() << " is " << dDims << " but "
          << sfadSizeName << " is " << static_size << "!\n";
      ss2 << " - Rebuild with " << sfadSizeName << "=" << dDims << "\n";
      if (static_size > dDims)
        *Teuchos::VerboseObjectBase::getDefaultOStream()
            << "WARNING: " << ss1.str()
            << "Continuing with this size may cause issues...\n" << ss2.str();
      else
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, ss1.str() + ss2.str());
    }
  }
}

} // namespace

namespace PHAL {

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian> (
  const Albany::Application* app, const Albany::MeshSpecsStruct* ms, bool responseEvaluation)
{
  int dDims = app->getNumEquations() * ms->ctd.node_count;
  const Teuchos::RCP<const Teuchos::ParameterList> pl = app->getProblemPL();
  if (Teuchos::nonnull(pl)) {
    const bool landIceCoupledFOH3D = !responseEvaluation && pl->get<std::string>("Name") == "LandIce Coupled FO H 3D";
    bool extrudedColumnCoupled = (responseEvaluation && pl->isParameter("Extruded Column Coupled in 2D Response")) ?
        pl->get<bool>("Extruded Column Coupled in 2D Response") : false;
    if (pl->isParameter("Extruded Column Coupled in 2D Residual")) {
      extrudedColumnCoupled |= pl->get<bool>("Extruded Column Coupled in 2D Residual");
    }
    if(landIceCoupledFOH3D || extrudedColumnCoupled)
      { //all column is coupled
        int side_node_count = ms->ctd.side[3].topology->node_count;
        int node_count = ms->ctd.node_count;
        int numLevels = app->getDiscretization()->getLayeredMeshNumberingGO()->numLayers+1;
        dDims = app->getNumEquations()*(node_count + side_node_count*numLevels);
      }
  }
  checkDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(dDims);
  return dDims;
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::Tangent> (
  const Albany::Application* app, const Albany::MeshSpecsStruct* /* ms */, bool /* responseEvaluation */)
{
  const int dDims = app->getTangentDerivDimension();
  checkDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(dDims);
  return dDims;
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv> (
  const Albany::Application* /* app */, const Albany::MeshSpecsStruct* ms, bool /* responseEvaluation */)
{
  //Mauro: currently distributed derivatives work only with scalar parameters, to be updated.
  const int dDims = ms->ctd.node_count;
  checkDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(dDims);
  return dDims;
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::HessianVec> (
  const Albany::Application* app, const Albany::MeshSpecsStruct* ms, bool responseEvaluation)
{
  const int derivativeDimension_x = getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(app, ms, responseEvaluation);
  const int derivativeDimension_p_dist = getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(app, ms, responseEvaluation);
  const int derivativeDimension_p_scal = getDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(app, ms, responseEvaluation);
  const int derivativeDimension_p_max = derivativeDimension_p_dist > derivativeDimension_p_scal ? derivativeDimension_p_dist : derivativeDimension_p_scal;
  const int derivativeDimension_max = derivativeDimension_x > derivativeDimension_p_max ? derivativeDimension_x : derivativeDimension_p_max;
  checkDerivativeDimensions<PHAL::AlbanyTraits::HessianVec>(derivativeDimension_max);
  return derivativeDimension_max;
}

template <typename EvalT>
int getDerivativeDimensions(const Albany::Application* app, const int ebi)
{
  return getDerivativeDimensions<EvalT>(app, app->getEnrichedMeshSpecs()[ebi].get());
}


namespace {
template<typename ScalarT>
struct A2V {
  std::vector<ScalarT>& v;
  A2V (std::vector<ScalarT>& v) : v(v) {}
  void operator() (typename Ref<const ScalarT>::type a, const int i) {
    v[i] = a;
  }
};

template<typename ScalarT>
struct V2A {
  const std::vector<ScalarT>& v;
  V2A (const std::vector<ScalarT>& v) : v(v) {}
  void operator() (typename Ref<ScalarT>::type a, const int i) {
    a = v[i];
  }
};

template<typename ScalarT>
void copy (const PHX::MDField<ScalarT>& a, std::vector<ScalarT>& v) {
  v.resize(a.size());
  A2V<ScalarT> a2v(v);
  loop(a2v, a);
}

template<typename ScalarT>
void copy (const std::vector<ScalarT>& v, PHX::MDField<ScalarT>& a) {
  V2A<ScalarT> v2a(v);
  loop(v2a, a);
}

} // namespace

template<typename ScalarT>
void reduceAll (
  const Teuchos_Comm& comm, const Teuchos::EReductionType reduct_type,
  PHX::MDField<ScalarT>& a)
{
  Kokkos::DynRankView<ScalarT,Albany::DevLayout,PHX::Device> v(a.get_view());
  Kokkos::deep_copy(v, a.get_view());
  Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, static_cast<int>(a.get_view().size()), a.get_view().data(), v.data());
  Kokkos::deep_copy(a.get_view(), v);
}

template<typename ScalarT>
void reduceAll (
  const Teuchos_Comm& comm, const Teuchos::EReductionType reduct_type,
  ScalarT& a)
{
  ScalarT b = a;
  Teuchos::reduceAll(comm, reduct_type, 1, &a, &b);
  a = b;
}

template<typename ScalarT>
void broadcast (const Teuchos_Comm& comm, const int root_rank,
                PHX::MDField<ScalarT>& a) {
  std::vector<ScalarT> v;
  copy<ScalarT>(a, v);
  Teuchos::broadcast<int, ScalarT>(comm, root_rank, v.size(), &v[0]);
  copy<ScalarT>(v, a);
}

template int getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
    const Albany::Application*, const int);
template int getDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(
    const Albany::Application*, const int);
template int getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(
    const Albany::Application*, const int);
template int getDerivativeDimensions<PHAL::AlbanyTraits::HessianVec>(
    const Albany::Application*, const int);

#  ifdef ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
#define apply_to_all_ad_types(macro)            \
  macro(RealType)                               \
  macro(FadType)                                \
  macro(TanFadType)                             \
  macro(HessianVecFad)
#  else
#define apply_to_all_ad_types(macro)            \
  macro(RealType)                               \
  macro(FadType)                                \
  macro(HessianVecFad)
#  endif

#define eti(T)                                                              \
  template void reduceAll<T> (                                              \
    const Teuchos_Comm&, const Teuchos::EReductionType, PHX::MDField<T>&);  \
  template void reduceAll<T> (                                              \
    const Teuchos_Comm&, const Teuchos::EReductionType, T&);                \
  template void broadcast<T> (                                              \
    const Teuchos_Comm&, const int, PHX::MDField<T>&);
apply_to_all_ad_types(eti)
#undef eti
#undef apply_to_all_ad_types

} // namespace PHAL
