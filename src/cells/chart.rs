use nalgebra::{Point, RealField};

/// A chart mapping points in the physical domain to the parametric domain.
pub trait Chart<T: RealField, X, const M: usize> {
    /// Evaluates the parametrization of this chart, i.e. the inverse mapping.
    fn eval(&self, x: X) -> Point<T, M>;
}