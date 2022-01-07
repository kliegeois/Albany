#include "PyAlbany_Stokhos.hpp"

using namespace PyAlbany;

KLExpention::KLExpention(int _ndim) : ndim(_ndim) {
    domain_upper = Teuchos::Array<double>(ndim, 1.0);
    domain_lower = Teuchos::Array<double>(ndim, 0.0);
    correlation_lengths = Teuchos::Array<double>(ndim, 1.0);
    num_KL_per_dim = Teuchos::Array<int>(ndim, 1);

    eps = 1e-6;
    tol = 1e-10;
    max_it = 100;
    num_KL_terms = 1;
}

void KLExpention::setUpperBound(int i, double _domain_upper_i) {
    domain_upper[i] = _domain_upper_i;
}
void KLExpention::setLowerBound(int i, double _domain_lower_i) {
    domain_lower[i] = _domain_lower_i;
}
void KLExpention::setCorrelationLength(int i, double _correlation_length_i) {
    correlation_lengths[i] = _correlation_length_i;
}
void KLExpention::setNumberOfKLTerm(int i, double _num_KL_terms_i) {
    num_KL_per_dim[i] = _num_KL_terms_i;
}

void KLExpention::createModes() {
    Teuchos::ParameterList solverParams;
    solverParams.set("Number of KL Terms", num_KL_terms);
    solverParams.set("Mean", RST::zero());
    solverParams.set("Standard Deviation", RST::one());
    solverParams.set("Bound Perturbation Size", eps);
    solverParams.set("Nonlinear Solver Tolerance", tol);
    solverParams.set("Maximum Nonlinear Solver Iterations", max_it);

    solverParams.set("Domain Upper Bounds", domain_upper);
    solverParams.set("Domain Lower Bounds", domain_lower);
    solverParams.set("Correlation Lengths", correlation_lengths);
    solverParams.set("Number of KL Terms per dimension", num_KL_per_dim);

    randomField = RandomFieldType(solverParams);
}

Teuchos::RCP<PyTrilinosVector> KLExpention::getMode(int i_mode, Teuchos::RCP<PyTrilinosVector> x, Teuchos::RCP<PyTrilinosVector> y, Teuchos::RCP<PyTrilinosVector> z) {
    Teuchos::RCP<PyTrilinosVector> phi = Teuchos::rcp(new PyTrilinosVector(x->getMap()));
    auto phi_view = phi->getLocalView<PyTrilinosVector::node_type::device_type>();

    Kokkos::View<double *, Kokkos::LayoutLeft, PyTrilinosVector::node_type::device_type> weights("w", num_KL_terms);
    for (std::size_t j_mode = 0; j_mode<num_KL_terms; ++ j_mode) {
        weights(j_mode) = i_mode == j_mode ? 1 : 0;
    }

    if (ndim == 1) {
        auto x_view = x->getLocalView<PyTrilinosVector::node_type::device_type>();

        for (std::size_t i_node = 0; i_node<phi_view.extent(0); ++ i_node) {
            const double point[1] = {x_view(i_node, 0)};
            phi_view(i_node, 0) = randomField.evaluate(point, weights);            
        }
    }
    if (ndim == 2) {
        auto x_view = x->getLocalView<PyTrilinosVector::node_type::device_type>();
        auto y_view = y->getLocalView<PyTrilinosVector::node_type::device_type>();

        for (std::size_t i_node = 0; i_node<phi_view.extent(0); ++ i_node) {
            const double point[2] = {x_view(i_node, 0), y_view(i_node, 0)};
            phi_view(i_node, 0) = randomField.evaluate(point, weights);            
        }
    }
    if (ndim == 3) {
        auto x_view = x->getLocalView<PyTrilinosVector::node_type::device_type>();
        auto y_view = y->getLocalView<PyTrilinosVector::node_type::device_type>();
        auto z_view = z->getLocalView<PyTrilinosVector::node_type::device_type>();

        for (std::size_t i_node = 0; i_node<phi_view.extent(0); ++ i_node) {
            const double point[3] = {x_view(i_node, 0), y_view(i_node, 0), z_view(i_node, 0)};
            phi_view(i_node, 0) = randomField.evaluate(point, weights);            
        }
    }

    return phi;
}

