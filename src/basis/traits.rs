use nalgebra::{Const, Dyn, OMatrix, RealField};
use crate::basis::local::LocalBasis;

/// Set of basis functions.
///
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
/// - [`N`] : Number of components of the basis functions.
///   For scalar valued functions equal to `1`,
///   for vector valued functions equal to the dimension of the parametric domain.
pub trait Basis<T: RealField, X, const N: usize> {
    // todo: maybe move N to associated type NumComponents
    /// Returns the number of basis functions in this set.
    fn num_basis(&self) -> usize;

    /// Evaluates all basis functions at the parametric point `x`
    /// as the column-wise matrix `(b[1],...,b[n])`.
    fn eval(&self, x: X) -> OMatrix<T, Const<N>, Dyn>;
}

/// Basis functions for `H(grad)`-conforming spaces (i.e. nodal functions).
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
/// - [`D`] : Dimension of the reference domain.
pub trait HgradBasis<T: RealField, X, const D: usize> : LocalBasis<T, X, 1> {
    /// Evaluates the gradients of all basis functions at the parametric point `x` 
    /// as the column-wise matrix `(grad b[1],...,grad b[n])`.
    fn eval_grad(&self, x: X) -> OMatrix<T, Const<D>, Dyn>;
}