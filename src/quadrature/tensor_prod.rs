use crate::cells::cartesian::CartCell;
use crate::cells::geo::Cell;
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

impl<T> Quadrature<T, (T, T), UnitCube<2>> for MultiProd<T, GaussLegendre, 2>
    where T: RealField + Sum + Product + Copy,
{
    fn nodes_elem(&self, _elem: &UnitCube<2>) -> impl Iterator<Item=(T, T)> {
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

impl<T, const D: usize> Quadrature<T, [T; D], SymmetricUnitCube<D>> for MultiProd<T, GaussLegendre, D>
    where T: RealField + Sum + Product + Copy,
{
    fn nodes_elem(&self, _elem: &SymmetricUnitCube<D>) -> impl Iterator<Item=[T; D]> {
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

impl<T, const D: usize> Quadrature<T, [T; D], UnitCube<D>> for MultiProd<T, GaussLegendre, D>
    where T: RealField + Sum + Product + Copy,
{

    fn nodes_elem(&self, _elem: &UnitCube<D>) -> impl Iterator<Item=[T; D]> {
        self.quads.iter()
            .map(|quad| quad.nodes_elem(&UnitCube).collect_vec())
            .multi_cartesian_product()
            .map(|vec| vec.try_into().unwrap())
    }

    fn weights_elem(&self, _elem: &UnitCube<D>) -> impl Iterator<Item=T> {
        self.quads.iter()
            .map(|quad| quad.weights_elem(&UnitCube).collect_vec())
            .multi_cartesian_product()
            .map(|vec| vec.into_iter().product())
    }
}

impl <T, const D: usize> Quadrature<T, [T; D], CartCell<T, D>> for MultiProd<T, GaussLegendre, D>
    where T: RealField + Sum + Product + Copy,
{
    fn nodes_elem(&self, elem: &CartCell<T, D>) -> impl Iterator<Item=[T; D]> {
        let lerp = elem.geo_map();
        self.nodes_elem(&SymmetricUnitCube)
            .map(move |xi| lerp.transform_symmetric(Vector::from(xi)).into_arr())
    }

    fn weights_elem(&self, elem: &CartCell<T, D>) -> impl Iterator<Item=T> {
        // todo: use weights_elem on ref domain for this impl
        zip(&self.quads, elem.intervals())
            .map(|(quad, interval)| quad.weights_elem(&interval).collect_vec())
            .multi_cartesian_product()
            .map(|vec| vec.into_iter().product())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::point;
    use super::*;

    fn setup() -> GaussLegendreMulti<f64, 2> {
        GaussLegendreMulti {
            quads: [GaussLegendre::new(2).unwrap(), GaussLegendre::new(4).unwrap()],
            _phantom: Default::default(),
        }
    }

    #[test]
    fn nodes_elem() {
        let q = setup();

        // Test symmetric unit square [-1,1]^2
        let nodes: Vec<[f64; 2]> = q.nodes_elem(&SymmetricUnitCube).collect();
        assert_abs_diff_eq!(nodes[0].as_slice(), [0.57735, 0.861136].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[1].as_slice(), [0.57735, 0.339981].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[2].as_slice(), [0.57735, -0.339981].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[3].as_slice(), [0.57735, -0.861136].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[4].as_slice(), [-0.57735, 0.861136].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[5].as_slice(), [-0.57735, 0.339981].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[6].as_slice(), [-0.57735, -0.339981].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[7].as_slice(), [-0.57735, -0.861136].as_slice(), epsilon = 1e-5);

        // Test unit square [0,1]^2
        let nodes = <GaussLegendreMulti<f64, 2> as Quadrature<f64, [f64; 2], UnitCube<2>>>::nodes_elem(&q, &UnitCube::<2>).collect_vec();
        assert_abs_diff_eq!(nodes[0].as_slice(), [0.788675, 0.930568].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[1].as_slice(), [0.788675, 0.6699905].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[2].as_slice(), [0.788675, 0.3300095].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[3].as_slice(), [0.788675, 0.069432].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[4].as_slice(), [0.211325, 0.930568].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[5].as_slice(), [0.211325, 0.6699905].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[6].as_slice(), [0.211325, 0.3300095].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[7].as_slice(), [0.211325, 0.069432].as_slice(), epsilon = 1e-5);

        // Test rectangle [0,1]x[0.5,1]
        let cell = CartCell::new(point![0.0, 0.5], point![1.0, 1.0]);
        let nodes: Vec<[f64; 2]> = q.nodes_elem(&cell).collect();
        assert_abs_diff_eq!(nodes[0].as_slice(), [0.788675, 0.965284].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[1].as_slice(), [0.788675, 0.83499525].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[2].as_slice(), [0.788675, 0.66500475].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[3].as_slice(), [0.788675, 0.534716].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[4].as_slice(), [0.211325, 0.965284].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[5].as_slice(), [0.211325, 0.83499525].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[6].as_slice(), [0.211325, 0.66500475].as_slice(), epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[7].as_slice(), [0.211325, 0.534716].as_slice(), epsilon = 1e-5);
    }

    #[test]
    fn weights_elem() {
        let q = setup();

        // Test symmetric unit square [-1,1]^2
        let weights: Vec<f64> = q.weights_elem(&SymmetricUnitCube).collect();
        assert_abs_diff_eq!(weights[0], 0.347855, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[1], 0.652145, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[2], 0.652145, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[3], 0.347855, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[4], 0.347855, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[5], 0.652145, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[6], 0.652145, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[7], 0.347855, epsilon = 1e-5);

        // Test unit square [0,1]^2
        let weights = <GaussLegendreMulti<f64, 2> as Quadrature<f64, [f64; 2], UnitCube<2>>>::weights_elem(&q, &UnitCube::<2>).collect_vec();
        assert_abs_diff_eq!(weights[0], 0.08696375, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[1], 0.16303625, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[2], 0.16303625, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[3], 0.08696375, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[4], 0.08696375, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[5], 0.16303625, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[6], 0.16303625, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[7], 0.08696375, epsilon = 1e-5);

        // Test rectangle [0,1]x[0.5,1]
        let cell = CartCell::new(point![0.0, 0.5], point![1.0, 1.0]);
        let weights: Vec<f64> = q.weights_elem(&cell).collect();
        assert_abs_diff_eq!(weights[0], 0.043481875, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[1], 0.081518125, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[2], 0.081518125, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[3], 0.043481875, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[4], 0.043481875, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[5], 0.081518125, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[6], 0.081518125, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[7], 0.043481875, epsilon = 1e-5);
    }
}