//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_UNIVARIATEDISTRIBUTION_HPP
#define ALBANY_UNIVARIATEDISTRIBUTION_HPP

#include <boost/math/special_functions/erf.hpp>

namespace Albany
{

  class UnivariatDistribution
  {
    public:
      UnivariatDistribution()
      {
        // Nothing to be done here
      }
      UnivariatDistribution(Teuchos::ParameterList &distributionParams)
      {
        // Nothing to be done here
      }
      virtual double cdf(const double x);
      virtual double cdf_dx(const double x);
      virtual double cdf_dx_dx(const double x);

      virtual double ppf(const double x);
      virtual double ppf_dx(const double x);
      virtual double ppf_dx_dx(const double x);

      virtual double toNormalMapping(const double x);
      virtual double toNormalMapping_dx(const double x);
      virtual double toNormalMapping_dx_dx(const double x);
      virtual double fromNormalMapping(const double x);
      virtual double fromNormalMapping_dx(const double x);
      virtual double fromNormalMapping_dx_dx(const double x);
  };

  class NormalDistribution : public UnivariatDistribution
  {
    public:
      NormalDistribution() {
        mu = 0.;
        sigma = 1.;
      }
      NormalDistribution(Teuchos::ParameterList &distributionParams) {
        mu = distributionParams.get<double>("Loc", 0.);
        sigma = distributionParams.get<double>("Scale", 1.);
      }
      NormalDistribution(double mu_, double sigma_) {
        mu = mu_;
        sigma = sigma_;
      }
    
      double cdf(const double x) {
        return 0.5*(1+erf((x-mu)/(sigma*sqrt(2.))));
      }

      double cdf_dx(const double x) {
        return exp(-pow(x-mu,2)/(2*pow(sigma,2)))/(sqrt(2*M_PI)*sigma);
      }

      double cdf_dx_dx(const double x) {
        return -exp(-pow(x-mu,2)/(2*pow(sigma,2)))*(x-mu)/(sqrt(2*M_PI)*pow(sigma,3));
      }
      
      double ppf(const double x) {
        return mu + sigma * sqrt(2) * boost::math::erf_inv(2*x-1);
      }

      double ppf_dx(const double x) {
        return sqrt(2*M_PI) * sigma * exp(pow(boost::math::erf_inv(-1+2*x),2));
      }

      double ppf_dx_dx(const double x) {
        return 2*sqrt(2)*M_PI * sigma * exp(2*pow(boost::math::erf_inv(-1+2*x),2))*boost::math::erf_inv(2*x-1);
      }

      double toNormalMapping(const double x) {
        NormalDistribution standard(0, 1);
        return standard.ppf(this->cdf(x));        
      }

      double toNormalMapping_dx(const double x) {
        NormalDistribution standard(0, 1);
        return standard.ppf_dx(this->cdf(x)) * this->cdf_dx(x);
      }

      double toNormalMapping_dx_dx(const double x) {
        NormalDistribution standard(0, 1);
        return standard.ppf_dx_dx(this->cdf(x)) * pow(this->cdf_dx(x), 2) + standard.ppf_dx(this->cdf(x)) * this->cdf_dx_dx(x);
      }

      double fromNormalMapping(const double x) {
        NormalDistribution standard(0, 1);
        return this->ppf(standard.cdf(x));        
      }

      double fromNormalMapping_dx(const double x) {
        NormalDistribution standard(0, 1);
        return this->ppf_dx(standard.cdf(x)) * standard.cdf_dx(x);
      }

      double fromNormalMapping_dx_dx(const double x) {
        NormalDistribution standard(0, 1);
        return this->ppf_dx_dx(standard.cdf(x)) * pow(standard.cdf_dx(x), 2) + this->ppf_dx(this->cdf(x)) * standard.cdf_dx_dx(x);
      }
  
    private:
      double sigma, mu;
  };

  class LogNormalDistribution : public UnivariatDistribution
  {
    public:
      LogNormalDistribution(Teuchos::ParameterList &distributionParams) {
        mu = 0.;
        sigma = distributionParams.get<double>("Scale", 1.);
      }
      LogNormalDistribution(double sigma_) {
        mu = 0.;
        sigma = sigma_;
      }
    
      double cdf(const double x) {
        return 0.5*(1+erf(pow(log(x)-mu, 2)/(sigma*sqrt(2.))));
      }

      double cdf_dx(const double x) {
        return exp(-pow(log(x)-mu,2)/(2*pow(sigma,2)))/(sqrt(2*M_PI)*sigma*x);
      }

      double cdf_dx_dx(const double x) {
        return -exp(-pow(log(x)-mu,2)/(2*pow(sigma,2)))/(sqrt(2*M_PI)*sigma*pow(x,2)) - 2 * ((log(x)-mu)*exp(-pow(log(x)-mu,2)/(2*pow(sigma,2))))/(2*sqrt(2*M_PI)*pow(sigma,3)*pow(x,2));
      }
      
      double ppf(const double x) {
        return exp(mu + sqrt(2) * sigma * boost::math::erf_inv(2*x-1));
      }

      double ppf_dx(const double x) {
        return sqrt(2*M_PI)*sigma*exp(sqrt(2)*sigma*boost::math::erf_inv(2*x-1)+pow(boost::math::erf_inv(2*x-1),2)+mu);
      }

      double ppf_dx_dx(const double x) {
        return sqrt(2)*M_PI*sigma*(2*boost::math::erf_inv(2*x-1)+sqrt(2)*sigma) * exp(sqrt(2)*sigma*boost::math::erf_inv(2*x-1)+2*pow(boost::math::erf_inv(2*x-1),2)+mu);
      }

      double toNormalMapping(const double x) {
        NormalDistribution standard(0, 1);
        return standard.ppf(this->cdf(x));        
      }

      double toNormalMapping_dx(const double x) {
        NormalDistribution standard(0, 1);
        return standard.ppf_dx(this->cdf(x)) * this->cdf_dx(x);
      }

      double toNormalMapping_dx_dx(const double x) {
        NormalDistribution standard(0, 1);
        return standard.ppf_dx_dx(this->cdf(x)) * pow(this->cdf_dx(x), 2) + standard.ppf_dx(this->cdf(x)) * this->cdf_dx_dx(x);
      }

      double fromNormalMapping(const double x) {
        NormalDistribution standard(0, 1);
        return this->ppf(standard.cdf(x));        
      }

      double fromNormalMapping_dx(const double x) {
        NormalDistribution standard(0, 1);
        return this->ppf_dx(standard.cdf(x)) * standard.cdf_dx(x);
      }

      double fromNormalMapping_dx_dx(const double x) {
        NormalDistribution standard(0, 1);
        return this->ppf_dx_dx(standard.cdf(x)) * pow(standard.cdf_dx(x), 2) + this->ppf_dx(this->cdf(x)) * standard.cdf_dx_dx(x);
      }
  
    private:
      double mu, sigma;
  };
  
  class UniformDistribution : public UnivariatDistribution
  {
    public:
      UniformDistribution(Teuchos::ParameterList &distributionParams) {
        a = distributionParams.get<double>("Loc", 0.);
        b = distributionParams.get<double>("Scale", 1.) + a;
      }
      UniformDistribution(double a_, double b_) {
        a = a_;
        b = b_;
      }
    
      double cdf(const double x) {
        return (x-a)/(b-a);
      }

      double cdf_dx(const double x) {
        return 1./(b-a);
      }

      double cdf_dx_dx(const double x) {
        return 0;
      }
      
      double ppf(const double x) {
        return (b-a) * x + a;
      }

      double ppf_dx(const double x) {
        return (b-a);
      }

      double ppf_dx_dx(const double x) {
        return 0;
      }

      double toNormalMapping(const double x) {
        NormalDistribution standard(0, 1);
        return standard.ppf(this->cdf(x));        
      }

      double toNormalMapping_dx(const double x) {
        NormalDistribution standard(0, 1);
        return standard.ppf_dx(this->cdf(x)) * this->cdf_dx(x);
      }

      double toNormalMapping_dx_dx(const double x) {
        NormalDistribution standard(0, 1);
        return standard.ppf_dx_dx(this->cdf(x)) * pow(this->cdf_dx(x), 2) + standard.ppf_dx(this->cdf(x)) * this->cdf_dx_dx(x);
      }

      double fromNormalMapping(const double x) {
        NormalDistribution standard(0, 1);
        return this->ppf(standard.cdf(x));        
      }

      double fromNormalMapping_dx(const double x) {
        NormalDistribution standard(0, 1);
        return this->ppf_dx(standard.cdf(x)) * standard.cdf_dx(x);
      }

      double fromNormalMapping_dx_dx(const double x) {
        NormalDistribution standard(0, 1);
        return this->ppf_dx_dx(standard.cdf(x)) * pow(standard.cdf_dx(x), 2) + this->ppf_dx(this->cdf(x)) * standard.cdf_dx_dx(x);
      }
      
    private:
      double a, b;
  };

} // namespace Albany

#endif // ALBANY_UNIVARIATEDISTRIBUTION_HPP
