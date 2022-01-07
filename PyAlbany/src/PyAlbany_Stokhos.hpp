//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PYALBANY_STOKHOS_H
#define PYALBANY_STOKHOS_H

#include "Stokhos_KL_ExponentialRandomField.hpp"
#include "Albany_PyAlbanyTypes.hpp"

namespace PyAlbany
{
    class KLExpention
    {
    private:
#ifndef SWIG
        typedef Stokhos::KL::ExponentialRandomField<double> RandomFieldType;
        typedef Teuchos::ScalarTraits<double> RST;

        const int ndim;
        Teuchos::Array<double> domain_upper, domain_lower, correlation_lengths;
        Teuchos::Array<int> num_KL_per_dim;
        double eps, tol;
        int max_it, num_KL_terms;
        RandomFieldType randomField;
#endif        

    public:
        KLExpention(int _ndim = 2);

        void setBoundPerturbationSize(double _eps) {eps = _eps;};
        void setNonlinearSolverTolerance(double _tol) {tol = _tol;};
        void setMaximumNonlinearSolverIterations(int _max_it) {max_it = _max_it;};
        void setNumberOfKLTerms(int _num_KL_terms) {num_KL_terms = _num_KL_terms;};

        void setUpperBound(int i, double _domain_upper_i);
        void setLowerBound(int i, double _domain_lower_i);
        void setCorrelationLength(int i, double _correlation_length_i);
        void setNumberOfKLTerm(int i, double _num_KL_terms_i);

        void createModes();

        Teuchos::RCP<PyTrilinosVector> getMode(int i, Teuchos::RCP<PyTrilinosVector> x, Teuchos::RCP<PyTrilinosVector> y = Teuchos::null, Teuchos::RCP<PyTrilinosVector> z = Teuchos::null);

        ~KLExpention(){}
    };
}

#endif // PYALBANY_STOKHOS_H
