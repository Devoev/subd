use crate::cells::geo::Cell;
use crate::cells::hyper_rectangle::HyperRectangle;
use crate::quadrature::traits::{Quadrature, RefQuadrature};
use gauss_quad::GaussLegendre;
use itertools::Itertools;
use nalgebra::{Point, RealField, Vector};
use std::iter::{zip, Product, Sum};
use std::marker::PhantomData;

/// Quadrature rule on tensor-product domains.
#[derive(Clone, Copy, Debug)]
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
    where T: RealField + Sum,
          Q: RefQuadrature<T, Node=T>
{
    /// Constructs a new [`MultiProd`] from the given `D` quadrature rules per parametric direction.
    pub fn new(quads: [Q; D]) -> Self {
        MultiProd { quads, _phantom: PhantomData }
    }
}

impl <T: RealField + Sum, const D: usize> GaussLegendreMulti<T, D> {
    /// Constructs a new [`GaussLegendreMulti`] with the given `degrees` per parametric direction.
    pub fn with_degrees(degrees: [usize; D]) -> Self {
        GaussLegendreMulti::new(degrees.map(|degree| GaussLegendre::new(degree).unwrap()))
    }
}


// todo: make this generic over Q again

impl<T, const D: usize> RefQuadrature<T> for MultiProd<T, GaussLegendre, D>
    where T: RealField + Sum + Product + Copy,
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

impl <T, const D: usize> Quadrature<T, D> for MultiProd<T, GaussLegendre, D>
    where T: RealField + Sum + Product + Copy,
{
    type Elem = HyperRectangle<T, D>;

    fn nodes_elem(&self, elem: &Self::Elem) -> impl Iterator<Item=Point<T, D>> {
        let lerp = elem.geo_map();
        self.nodes_ref()
            .map(move |xi| lerp.transform_symmetric(Vector::from(xi)))
    }

    fn weights_elem(&self, elem: &Self::Elem) -> impl Iterator<Item=T> {
        zip(&self.quads, elem.intervals())
            .map(|(quad, interval)| quad.weights_elem(&interval).collect_vec())
            .multi_cartesian_product()
            .map(|vec| vec.into_iter().product())
    }
}