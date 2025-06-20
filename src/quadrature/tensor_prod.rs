use crate::cells::geo::Cell;
use crate::cells::hyper_rectangle::HyperRectangle;
use crate::cells::unit_cube::{SymmetricUnitCube, UnitCube};
use crate::index::dimensioned::Dimensioned;
use crate::quadrature::traits::Quadrature;
use gauss_quad::GaussLegendre;
use itertools::Itertools;
use nalgebra::{RealField, Vector};
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

impl<T, Q, const D: usize> MultiProd<T, Q, D>
    where T: RealField + Sum,
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

// todo: make this generic over Q and E again

impl<T> Quadrature<T, UnitCube<2>, 2> for MultiProd<T, GaussLegendre, 2>
    where T: RealField + Sum + Product + Copy,
{
    type Node = (T, T);

    fn nodes_elem(&self, _elem: &UnitCube<2>) -> impl Iterator<Item=Self::Node> {
        self.quads[0].nodes_elem(&UnitCube)
            .cartesian_product(self.quads[1].nodes_elem(&UnitCube).collect_vec())
    }

    fn weights_elem(&self, _elem: &UnitCube<2>) -> impl Iterator<Item=T> {
        self.quads[0].weights_elem(&UnitCube)
            .cartesian_product(self.quads[1].weights_elem(&UnitCube).collect_vec())
            .map(|(wi, wj): (T, T)| wi * wj)
    }
}

// todo: are all 3 implementations really needed? Or is the HyperCube one sufficient?
//  or just merge all into one generic

impl<T, const D: usize> Quadrature<T, SymmetricUnitCube<D>, D> for MultiProd<T, GaussLegendre, D>
    where T: RealField + Sum + Product + Copy,
{
    type Node = [T; D];

    fn nodes_elem(&self, _elem: &SymmetricUnitCube<D>) -> impl Iterator<Item=Self::Node> {
        self.quads.iter()
            .map(|quad| quad.nodes_elem(&SymmetricUnitCube).collect_vec())
            .multi_cartesian_product()
            .map(|vec| vec.try_into().unwrap())
    }

    fn weights_elem(&self, _elem: &SymmetricUnitCube<D>) -> impl Iterator<Item=T> {
        self.quads.iter()
            .map(|quad| quad.weights_elem(&SymmetricUnitCube).collect_vec())
            .multi_cartesian_product()
            .map(|vec| vec.into_iter().product())
    }
}

// impl<T, const D: usize> Quadrature<T, UnitCube<D>, D> for MultiProd<T, GaussLegendre, D>
// where T: RealField + Sum + Product + Copy,
// {
//     type Node = [T; D];
//
//     fn nodes_elem(&self, _elem: &UnitCube<D>) -> impl Iterator<Item=Self::Node> {
//         self.quads.iter()
//             .map(|quad| quad.nodes_elem(&UnitCube).collect_vec())
//             .multi_cartesian_product()
//             .map(|vec| vec.try_into().unwrap())
//     }
//
//     fn weights_elem(&self, _elem: &UnitCube<D>) -> impl Iterator<Item=T> {
//         self.quads.iter()
//             .map(|quad| quad.weights_elem(&UnitCube).collect_vec())
//             .multi_cartesian_product()
//             .map(|vec| vec.into_iter().product())
//     }
// }

impl <T, const D: usize> Quadrature<T, HyperRectangle<T, D>, D> for MultiProd<T, GaussLegendre, D>
    where T: RealField + Sum + Product + Copy,
{
    type Node = [T; D];

    fn nodes_elem(&self, elem: &HyperRectangle<T, D>) -> impl Iterator<Item=Self::Node> {
        let lerp = elem.geo_map();
        self.nodes_elem(&SymmetricUnitCube)
            .map(move |xi| lerp.transform_symmetric(Vector::from(xi)).into_arr())
    }

    fn weights_elem(&self, elem: &HyperRectangle<T, D>) -> impl Iterator<Item=T> {
        // todo: use weights_elem on ref domain for this impl
        zip(&self.quads, elem.intervals())
            .map(|(quad, interval)| quad.weights_elem(&interval).collect_vec())
            .multi_cartesian_product()
            .map(|vec| vec.into_iter().product())
    }
}