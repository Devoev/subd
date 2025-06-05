use nalgebra::SMatrix;
use crate::index::dimensioned::Dimensioned;

// todo: this is WIP. possibly change

/// A differentiable map from [`N`]-dimensional to [`M`]-dimensional coordinates `X → Y`.
pub trait DiffMap<T, X: Dimensioned<T, N>, Y: Dimensioned<T, M>, const N: usize, const M: usize> {
    /// The differential of this map.
    type Differential: Pushforward<T, X, N, M>;

    /// Evaluates the mapping at `x`.
    fn eval(&self, x: X) -> Y;

    /// Returns the differential of this map.
    fn differential(&self) -> Self::Differential;
}

/// Pushforward, or differential, of a [`DiffMap`].
pub trait Pushforward<T, X: Dimensioned<T, N>, const N: usize, const M: usize> {
    /// Evaluates the coordinate representation (Jacobian matrix) of the pushforward at `x`.
    fn eval(&self, x: X) -> SMatrix<T, M, N>;
}