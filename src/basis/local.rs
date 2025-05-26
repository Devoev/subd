use crate::basis::traits::Basis;
use crate::index::multi_index::MultiIndex;
use nalgebra::{Const, Dyn, OMatrix, RealField};
use crate::index::dimensioned::Strides;

/// Set of local basis functions.
///
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
/// - [`N`] : Number of components of the basis functions.
///   For scalar valued functions equal to `1`,
///   for vector valued functions equal to the dimension of the parametric domain.
pub trait LocalBasis<T: RealField, X, const D: usize, const N: usize> : Basis<T, X, N> {
    // todo: possibly change to IntoIterator or separate trait/ struct all together
    /// Global index type.
    type Index: MultiIndex<D, usize>;
    /// Iterator over global indices.
    type GlobalIndices: Iterator<Item = Self::Index>;
    
    /// Returns an iterator over all global indices the local basis functions
    /// in this set correspond to.
    fn global_indices(&self) -> Self::GlobalIndices;

    // todo: Create LinIndexIter struct or similar, that implements Clone if GlobalIndices does
    /// Returns an iterator over all global linear indices using the global `strides`.
    fn global_lin_indices(&self, strides: &Strides<D>) -> impl Iterator<Item = usize> {
        self.global_indices().map(|i| i.into_lin(strides))
    }
}

/// Local basis functions for `H(grad)`-conforming spaces (i.e. nodal functions).
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
/// - [`D`] : Dimension of the reference domain.
pub trait LocalHgradBasis<T: RealField, X, const D: usize> : LocalBasis<T, X, D, 1> {
    /// Evaluates the gradients of all local basis functions at the parametric point `x` 
    /// as the column-wise matrix `(grad b[1],...,grad b[n])`.
    fn eval_grad(&self, x: X) -> OMatrix<T, Const<D>, Dyn>;
}

// todo: maybe add eval_vals_and_grad function that evaluates both function values and gradients
//  it makes sense to use such a method for a more efficient implementation