use nalgebra::RealField;
use crate::basis::traits::{Basis, NumBasis};

/// Set of global basis functions which restricted to a single element form a [local basis][`Self::LocalBasis`].
///
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
/// - [`N`] : Number of components of the basis functions.
///   For scalar valued functions equal to `1`,
///   for vector valued functions equal to the dimension of the parametric domain.
pub trait GlobalToLocalBasis<T: RealField, X, const N: usize>: NumBasis {
    /// Element type.
    type Elem;
    
    /// Local basis for each [`Self::Elem`].
    type LocalBasis: Basis<T, X, N>;

    // todo: possibly change to IntoIterator or separate trait/ struct all together
    /// Iterator over linear global indices.
    type GlobalIndices: Iterator<Item = usize> + Clone;

    /// Returns the [`Self::LocalBasis`] for the given `elem`,
    /// i.e. the restriction of this basis to the element.
    fn local_basis(&self, elem: &Self::Elem) -> Self::LocalBasis;

    /// Returns an iterator over all global indices of the local basis of `elem`.
    fn global_indices(&self, local_basis: &Self::LocalBasis) -> Self::GlobalIndices;
}