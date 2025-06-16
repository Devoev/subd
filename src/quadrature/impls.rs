use crate::cells::geo::Cell;
use crate::cells::hyper_rectangle::HyperRectangle;
use crate::cells::lerp::Lerp;
use crate::quadrature::traits::{Quadrature, RefQuadrature};
use gauss_quad::GaussLegendre;
use nalgebra::{vector, Point, RealField, U1};
use std::iter::Sum;

impl <T: RealField + Sum> RefQuadrature<T> for GaussLegendre {
    type Node = T;

    fn nodes_ref(&self) -> impl Iterator<Item=Self::Node> {
        self.nodes()
            .map(|&xi| T::from_f64(xi).unwrap())
    }

    fn weights_ref(&self) -> impl Iterator<Item=T> {
        self.weights()
            .map(|&wi| T::from_f64(wi).unwrap())
    }
}

impl <T: RealField + Copy + Sum> Quadrature<T, 1> for GaussLegendre {
    type Elem = HyperRectangle<T, 1>;

    fn nodes_elem(&self, elem: &Self::Elem) -> impl Iterator<Item=Point<T, 1>> {
        let lerp: Lerp<T, 1> = <HyperRectangle<T, 1> as Cell<T, T, 1, 1>>::geo_map(elem);
        self.nodes_ref().map(move |xi: T| lerp.transform_symmetric(vector![xi]))
    }

    fn weights_elem(&self, elem: &Self::Elem) -> impl Iterator<Item=T> {
        // todo: implement this by giving Lerp a pushforward/ jacobian method
        let d_phi = (elem.b.x - elem.a.x) / T::from_i32(2).unwrap();
        self.weights_ref().map(move |wi: T| wi * d_phi)
    }
}