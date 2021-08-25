//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_LDTRATERESPONSEFUNCTION_HPP
#define ALBANY_LDTRATERESPONSEFUNCTION_HPP

#include "Albany_SamplingBasedScalarResponseFunction.hpp"
#include "Albany_Utils.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"

#include "Albany_UnivariateDistribution.hpp"

namespace Albany
{

  class Application;
  class CombineAndScatterManager;

  /*!
 * \brief Reponse function representing the average of the solution values
 */
  class LDTRateResponseFunction : public SamplingBasedScalarResponseFunction
  {
  public:
    //! Constructor
    LDTRateResponseFunction(const Teuchos::RCP<const Application> &app,
                            Teuchos::ParameterList &responseParams);

    //! Get the number of responses
    unsigned int numResponses() const;

    //! Setup response function
    void setup();

    //! Evaluate responses
    void evaluateResponse(const double current_time,
                          const Teuchos::RCP<const Thyra_Vector> &x,
                          const Teuchos::RCP<const Thyra_Vector> &xdot,
                          const Teuchos::RCP<const Thyra_Vector> &xdotdot,
                          const Teuchos::Array<ParamVec> &p,
                          const Teuchos::RCP<Thyra_Vector> &g);

    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    void evaluateTangent(const double alpha,
                         const double beta,
                         const double omega,
                         const double current_time,
                         bool sum_derivs,
                         const Teuchos::RCP<const Thyra_Vector> &x,
                         const Teuchos::RCP<const Thyra_Vector> &xdot,
                         const Teuchos::RCP<const Thyra_Vector> &xdotdot,
                         const Teuchos::Array<ParamVec> &p,
                         int parameter_index,
                         ParamVec *deriv_p,
                         const Teuchos::RCP<const Thyra_MultiVector> &Vx,
                         const Teuchos::RCP<const Thyra_MultiVector> &Vxdot,
                         const Teuchos::RCP<const Thyra_MultiVector> &Vxdotdot,
                         const Teuchos::RCP<const Thyra_MultiVector> &Vp,
                         const Teuchos::RCP<Thyra_Vector> &g,
                         const Teuchos::RCP<Thyra_MultiVector> &gx,
                         const Teuchos::RCP<Thyra_MultiVector> &gp);

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
    void evaluateGradient(const double current_time,
                          const Teuchos::RCP<const Thyra_Vector> &x,
                          const Teuchos::RCP<const Thyra_Vector> &xdot,
                          const Teuchos::RCP<const Thyra_Vector> &xdotdot,
                          const Teuchos::Array<ParamVec> &p,
                          ParamVec *deriv_p,
                          const Teuchos::RCP<Thyra_Vector> &g,
                          const Teuchos::RCP<Thyra_MultiVector> &dg_dx,
                          const Teuchos::RCP<Thyra_MultiVector> &dg_dxdot,
                          const Teuchos::RCP<Thyra_MultiVector> &dg_dxdotdot,
                          const Teuchos::RCP<Thyra_MultiVector> &dg_dp);

    //! Evaluate distributed parameter derivative dg/dp
    void evaluateDistParamDeriv(
        const double current_time,
        const Teuchos::RCP<const Thyra_Vector> &x,
        const Teuchos::RCP<const Thyra_Vector> &xdot,
        const Teuchos::RCP<const Thyra_Vector> &xdotdot,
        const Teuchos::Array<ParamVec> &param_array,
        const std::string &dist_param_name,
        const Teuchos::RCP<Thyra_MultiVector> &dg_dp);

    void evaluate_HessVecProd_xx(
        const double current_time,
        const Teuchos::RCP<const Thyra_MultiVector> &v,
        const Teuchos::RCP<const Thyra_Vector> &x,
        const Teuchos::RCP<const Thyra_Vector> &xdot,
        const Teuchos::RCP<const Thyra_Vector> &xdotdot,
        const Teuchos::Array<ParamVec> &param_array,
        const Teuchos::RCP<Thyra_MultiVector> &Hv_dp);

    void evaluate_HessVecProd_xp(
        const double current_time,
        const Teuchos::RCP<const Thyra_MultiVector> &v,
        const Teuchos::RCP<const Thyra_Vector> &x,
        const Teuchos::RCP<const Thyra_Vector> &xdot,
        const Teuchos::RCP<const Thyra_Vector> &xdotdot,
        const Teuchos::Array<ParamVec> &param_array,
        const std::string &dist_param_direction_name,
        const Teuchos::RCP<Thyra_MultiVector> &Hv_dp);

    void evaluate_HessVecProd_px(
        const double current_time,
        const Teuchos::RCP<const Thyra_MultiVector> &v,
        const Teuchos::RCP<const Thyra_Vector> &x,
        const Teuchos::RCP<const Thyra_Vector> &xdot,
        const Teuchos::RCP<const Thyra_Vector> &xdotdot,
        const Teuchos::Array<ParamVec> &param_array,
        const std::string &dist_param_name,
        const Teuchos::RCP<Thyra_MultiVector> &Hv_dp);

    void evaluate_HessVecProd_pp(
        const double current_time,
        const Teuchos::RCP<const Thyra_MultiVector> &v,
        const Teuchos::RCP<const Thyra_Vector> &x,
        const Teuchos::RCP<const Thyra_Vector> &xdot,
        const Teuchos::RCP<const Thyra_Vector> &xdotdot,
        const Teuchos::Array<ParamVec> &param_array,
        const std::string &dist_param_name,
        const std::string &dist_param_direction_name,
        const Teuchos::RCP<Thyra_MultiVector> &Hv_dp);

    virtual void
    printResponse(Teuchos::RCP<Teuchos::FancyOStream> out);

  private:
    void evaluateResponseImpl (const Teuchos::Array<ParamVec> &p,
                              Thyra_Vector& g);

    void evaluateGradientImpl (const Teuchos::Array<ParamVec> &p,
		                           Teuchos::SerialDenseVector<int, double>& dgdp);

    Teuchos::RCP<const Application> app_;

    Teuchos::RCP<Thyra_Vector> g_;
    int n_parameters;
    Teuchos::RCP<Teuchos::SerialDenseVector<int, double>> theta_0;
    Teuchos::RCP<Teuchos::SerialDenseMatrix<int, double>> C;
    Teuchos::RCP<Teuchos::SerialDenseMatrix<int, double>> invC;
  };

} // namespace Albany

#endif // ALBANY_LDTRATERESPONSEFUNCTION_HPP
