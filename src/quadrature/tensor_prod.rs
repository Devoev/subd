use std::marker::PhantomData;

/// Quadrature rule on tensor-product domains.
pub struct MultiProd<T, Q> {
    /// Quadrature nodes for a parametric direction.
    nodes: Vec<T>,
    /// Quadrature weights for a parametric direction.
    weights: Vec<T>,

    _phantoms: PhantomData<Q>,
}

// todo: implement tensor product quadrature as in TensorProdGaussLegendre
//  maybe include scale factor in weights and add weights function in trait