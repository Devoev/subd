use std::iter::{zip, Sum};
use gauss_quad::GaussLegendre;
use nalgebra::RealField;
use crate::quadrature::traits::RefQuadrature;

impl <T: RealField + Sum> RefQuadrature<T> for GaussLegendre {
    type Node = T;

    fn nodes_ref(&self) -> impl Iterator<Item=Self::Node> {
        self.nodes()
            .map(|&xi| T::from_f64((xi + 1.0) / 2.0).unwrap())
    }

    fn integrate_ref(&self, f: impl IntoIterator<Item=T>) -> T {
        let scale = T::from_f64(0.5).unwrap();
        zip(f, self.weights())
            .map(|(fi, &wi)| fi * T::from_f64(wi).unwrap())
            .sum::<T>() * scale
    }
}