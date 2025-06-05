use crate::index::dimensioned::Dimensioned;
use nalgebra::{Point, SMatrix, Scalar};

/// A chart mapping points in the physical domain to the parametric domain.
pub trait Chart<T: Scalar, X: Dimensioned<T, D>, const D: usize, const M: usize> {
    /// Evaluates the parametrization of this chart, i.e. the inverse mapping.
    fn eval(&self, x: X) -> Point<T, M>;
    
    /// Evaluates the coordinate representation (Jacobian matrix) 
    /// of the differential aka. pushforward at `x`.
    fn eval_diff(&self, x: X) -> SMatrix<T, M, D>;
}