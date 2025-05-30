use std::iter::{zip, Sum};
use gauss_quad::legendre::GaussLegendreError;
use nalgebra::RealField;
use num_traits::FromPrimitive;
use crate::quadrature::traits::Quadrature;

// todo: should this be implement over [0,1] instead?
/// Gauss-Legendre quadrature rule on the domain `[-1,1]`.
pub struct GaussLegendre<T> {
    /// Quadrature nodes.
    nodes: Vec<T>,
    /// Quadrature weights.
    weights: Vec<T>,
}

impl <T: FromPrimitive> GaussLegendre<T> {
    /// Scale factor for the weights.
    const SCALE_FACTOR: f64 = 0.5;
    
    /// Constructs a new Gauss-Legendre quadrature rule of the given `degree` by
    /// delegating the construction to [`gauss_quad::GaussLegendre`] of the `gauss_quad` crate.
    pub fn new(degree: usize) -> Result<Self, GaussLegendreError> {
        let quad = gauss_quad::GaussLegendre::new(degree)?;
        Ok(GaussLegendre {
            nodes: quad.nodes().map(|&n| T::from_f64(n).unwrap()).collect(),
            weights: quad.weights().map(|&w| T::from_f64(w).unwrap()).collect(),
        })
    }
}

impl <T: RealField + Copy + Sum> Quadrature<T> for GaussLegendre<T> {
    type Node = T;

    fn nodes(&self) -> impl Iterator<Item=Self::Node> {
        self.nodes.iter().cloned()
    }
    
    // todo: possibly implement this using dot product instead
    fn integrate(&self, f: impl IntoIterator<Item=T>) -> T {
        let scale = T::from_f64(Self::SCALE_FACTOR).unwrap();
        zip(f, &self.weights)
            .map(|(fi, &wi)| fi * wi)
            .sum::<T>() * scale
    }
}