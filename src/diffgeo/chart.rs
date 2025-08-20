use crate::index::dimensioned::Dimensioned;
use nalgebra::{Point, SMatrix, Scalar};

/// A chart mapping points in the physical domain to the parametric domain.
pub trait Chart<T: Scalar, const D: usize, const M: usize> {
    /// Coordinate of a parametric point.
    type Coord;
    
    /// Evaluates the parametrization of this chart, i.e. the inverse mapping.
    fn eval(&self, x: Self::Coord) -> Point<T, M>;
    
    /// Evaluates the coordinate representation (Jacobian matrix) 
    /// of the differential aka. pushforward at `x`.
    fn eval_diff(&self, x: Self::Coord) -> SMatrix<T, M, D>;
}