use std::iter::{zip, Sum};
use gauss_quad::GaussLegendre;
use nalgebra::RealField;
use crate::cells::hyper_rectangle::HyperRectangle;
use crate::quadrature::traits::{ElementQuadrature, RefQuadrature};

impl <T: RealField + Sum> RefQuadrature<T> for GaussLegendre {
    type Node = T;

    fn nodes_ref(&self) -> impl Iterator<Item=Self::Node> {
        self.nodes()
            .map(|&xi| T::from_f64((xi + 1.0) / 2.0).unwrap())
    }

    fn weights_ref(&self) -> impl Iterator<Item=T> {
        self.weights()
            .map(|&wi| T::from_f64(wi / 2.0).unwrap())
    }

    fn integrate_ref(&self, f: impl IntoIterator<Item=T>) -> T {
        let scale = T::from_f64(0.5).unwrap();
        zip(f, self.weights())
            .map(|(fi, &wi)| fi * T::from_f64(wi).unwrap())
            .sum::<T>() * scale
    }
}

impl <T: RealField + Copy + Sum> ElementQuadrature<T, 1> for GaussLegendre {
    type Elem = HyperRectangle<T, 1>;
}