//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_POWER_SCALAR_RESPONSE_FUNCTION_HPP
#define ALBANY_POWER_SCALAR_RESPONSE_FUNCTION_HPP

#include "Albany_SamplingBasedScalarResponseFunction.hpp"
#include "Teuchos_Array.hpp"

namespace Albany {

  /*!
   * \brief A response function that ...
   */
  class PowerScalarResponseFunction :
    public SamplingBasedScalarResponseFunction {
  public:
  
    //! Default constructor
    PowerScalarResponseFunction(
      const Teuchos::RCP<const Teuchos_Comm>& commT,
      const Teuchos::RCP<ScalarResponseFunction>& response,
      const double scalar_target = 0.,
      const double exponent = 2.);

    //! Setup response function
    virtual void setup();

    //!Perform post registration setup
    virtual void postRegSetup();

    //! Destructor
    virtual ~PowerScalarResponseFunction();

    //! Get the number of responses
    virtual unsigned int numResponses() const;

    //! Evaluate response
    virtual void 
    evaluateResponse(const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		  const Teuchos::Array<ParamVec>& p,
		  const Teuchos::RCP<Thyra_Vector>& g); 

    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    virtual void 
    evaluateTangent(const double alpha, 
		  const double beta,
		  const double omega,
		  const double current_time,
		  bool sum_derivs,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		  Teuchos::Array<ParamVec>& p,
      const int parameter_index,
      const Teuchos::RCP<const Thyra_MultiVector>& Vx,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vp,
      const Teuchos::RCP<Thyra_Vector>& g,
      const Teuchos::RCP<Thyra_MultiVector>& gx,
      const Teuchos::RCP<Thyra_MultiVector>& gp);
    
    virtual void 
    evaluateGradient(const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		  const Teuchos::Array<ParamVec>& p,
		  const int parameter_index,
		  const Teuchos::RCP<Thyra_Vector>& g,
		  const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
		  const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
		  const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
		  const Teuchos::RCP<Thyra_MultiVector>& dg_dp);

    virtual void
    printResponse(Teuchos::RCP<Teuchos::FancyOStream> out);

    void updateTarget(double target);
    void updateExponent(double exponent);

  private:

    //! Evaluate Multi Vector distributed derivative dg_dp
    virtual void
    evaluateDistParamDeriv(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
		  const Teuchos::RCP<Thyra_MultiVector>& dg_dp);

    virtual void
    evaluate_HessVecProd_xx(
      const double current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const Teuchos::RCP<Thyra_MultiVector>& Hv_dp);

    virtual void
    evaluate_HessVecProd_xp(
      const double current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_direction_name,
      const Teuchos::RCP<Thyra_MultiVector>& Hv_dp);

    virtual void
    evaluate_HessVecProd_px(
      const double current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      const Teuchos::RCP<Thyra_MultiVector>& Hv_dp);

    virtual void
    evaluate_HessVecProd_pp(
      const double current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      const std::string& dist_param_direction_name,
      const Teuchos::RCP<Thyra_MultiVector>& Hv_dp);

  private:

    //! Private to prohibit copying
    PowerScalarResponseFunction(const PowerScalarResponseFunction&);
    
    //! Private to prohibit copying
    PowerScalarResponseFunction& operator=(const PowerScalarResponseFunction&);

    Teuchos::RCP<Thyra_Vector> g_;
    Teuchos::RCP<Thyra_Vector> f_;

  protected:

    //! Response functions to add
    Teuchos::RCP<ScalarResponseFunction> response;
    double scalar_target;
    double exponent;

  };

} // namespace Albany

#endif // ALBANY_POWER_SCALAR_RESPONSE_FUNCTION_HPP
