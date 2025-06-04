use crate::index::dimensioned::Dimensioned;
use nalgebra::{Point, SMatrix, Scalar};

// todo: this is WIP. possibly change

/// A mapping from [`N`]-dimensional to [`M`]-dimensional coordinates `X → Y`.
pub trait Mapping<T, X: Dimensioned<T, N>, Y: Dimensioned<T, M>, const N: usize, const M: usize> {
    /// The differential of this map.
    type Differential: Pushforward<T, X, N, M>;

    /// Evaluates the mapping at `x`.
    fn eval(&self, x: X) -> Y;

    /// Returns the differential of this map.
    fn differential(&self) -> Self::Differential;
}

/// A chart mapping points in the physical domain to the parametric domain.
/// The inverse mapping, i.e. the parametrization, can be evaluated using [`Mapping::eval`].
pub trait Chart<T: Scalar, X: Dimensioned<T, D>, const D: usize, const M: usize>: Mapping<T, X, Point<T, M>, D, M> {}

/// Pushforward, or differential, of a [`Mapping`].
pub trait Pushforward<T, X: Dimensioned<T, N>, const N: usize, const M: usize> {
    /// Evaluates the coordinate representation (Jacobian matrix) of the pushforward at `x`.
    fn eval(&self, x: X) -> SMatrix<T, M, N>;
}