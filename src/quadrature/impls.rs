use crate::cells::geo::Cell;
use crate::cells::cartesian::CartCell;
use crate::cells::lerp::Lerp;
use crate::cells::unit_cube::{SymmetricUnitCube, UnitCube};
use crate::quadrature::traits::Quadrature;
use gauss_quad::GaussLegendre;
use nalgebra::{vector, RealField};
use std::iter::Sum;

// todo: are the two implementations really needed? or is the 2nd one sufficient

impl <T: RealField + Sum> Quadrature<T, T, SymmetricUnitCube<1>> for GaussLegendre {
    fn nodes_elem(&self, _elem: &SymmetricUnitCube<1>) -> impl Iterator<Item=T> {
        self.nodes()
            .map(|&xi| T::from_f64(xi).unwrap())
    }

    fn weights_elem(&self, _elem: &SymmetricUnitCube<1>) -> impl Iterator<Item=T> {
        self.weights()
            .map(|&wi| T::from_f64(wi).unwrap())
    }
}

impl <T: RealField + Sum> Quadrature<T, T, UnitCube<1>> for GaussLegendre {
    fn nodes_elem(&self, _elem: &UnitCube<1>) -> impl Iterator<Item=T> {
        self.nodes()
            .map(|&xi| T::from_f64((xi + 1.0) / 2.0).unwrap())
    }

    fn weights_elem(&self, _elem: &UnitCube<1>) -> impl Iterator<Item=T> {
        self.weights()
            .map(|&wi| T::from_f64(wi / 2.0).unwrap())
    }
}

impl <T: RealField + Copy + Sum> Quadrature<T, T, CartCell<T, 1>> for GaussLegendre {
    fn nodes_elem(&self, elem: &CartCell<T, 1>) -> impl Iterator<Item=T> {
        let lerp: Lerp<T, 1> = <CartCell<T, 1> as Cell<T, T, 1, 1>>::geo_map(elem);
        self.nodes_elem(&SymmetricUnitCube).map(move |xi: T| lerp.transform_symmetric(vector![xi]).x)
    }

    fn weights_elem(&self, elem: &CartCell<T, 1>) -> impl Iterator<Item=T> {
        let lerp: Lerp<T, 1> = <CartCell<T, 1> as Cell<T, T, 1, 1>>::geo_map(elem);
        let d_phi = lerp.jacobian().x / T::from_i32(2).unwrap(); // todo: add new jacobian method for symmetric interval
        self.weights_elem(&SymmetricUnitCube).map(move |wi: T| wi * d_phi)
    }
}