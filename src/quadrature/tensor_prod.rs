use crate::quadrature::traits::{ElementQuadrature, RefQuadrature};
use itertools::Itertools;
use nalgebra::RealField;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use gauss_quad::GaussLegendre;
use crate::cells::hyper_rectangle::HyperRectangle;

/// Quadrature rule on tensor-product domains.
pub struct MultiProd<T, Q, const D: usize> {
    /// Quadrature rules for each parametric direction.
    quads: [Q; D],

    _phantom: PhantomData<T>
}

/// [`D`]-variate Gauss-Legendre quadrature.
pub type GaussLegendreMulti<T, const D: usize> = MultiProd<T, GaussLegendre, D>;

// todo: implement tensor product quadrature as in TensorProdGaussLegendre
//  maybe include scale factor in weights and add weights function in trait

impl<T, Q, const D: usize> MultiProd<T, Q, D>
    where T: RealField + Sum + Product + Clone,
          Q: RefQuadrature<T, Node=T>
{
    /// Constructs a new [`MultiProd`] from the given `D` quadrature rules per parametric direction.
    pub fn new(quads: [Q; D]) -> Self {
        MultiProd { quads, _phantom: PhantomData }
    }
}

impl<T: RealField + Copy + Sum, Q, const D: usize> RefQuadrature<T> for MultiProd<T, Q, D>
    where T: RealField + Sum + Product + Clone,
          Q: RefQuadrature<T, Node=T>
{
    type Node = [T; D];

    fn nodes_ref(&self) -> impl Iterator<Item=Self::Node> {
        self.quads.iter()
            .map(|quad| quad.nodes_ref().collect_vec())
            .multi_cartesian_product()
            .map(|vec| vec.try_into().unwrap())
    }

    fn weights_ref(&self) -> impl Iterator<Item=T> {
        self.quads.iter()
            .map(|quad| quad.weights_ref().collect_vec())
            .multi_cartesian_product()
            .map(|vec| vec.into_iter().product())
    }
}

impl <T: RealField + Copy + Sum, Q, const D: usize> ElementQuadrature<T, D> for MultiProd<T, Q, D>
    where T: RealField + Sum + Product + Clone,
          Q: ElementQuadrature<T, 1, Node=T>
{
    type Elem = HyperRectangle<T, D>;
}