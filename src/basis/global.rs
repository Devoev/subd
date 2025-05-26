use crate::basis::local::LocalBasis;
use nalgebra::RealField;

/// Set of global basis functions which restricted to a single element form a [`LocalBasis`].
///
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
/// - [`N`] : Number of components of the basis functions.
///   For scalar valued functions equal to `1`,
///   for vector valued functions equal to the dimension of the parametric domain.
pub trait GlobalBasis<T: RealField, X, const N: usize> {
    /// Element type.
    type Elem;
    
    /// Local basis for each [`Self::Elem`].
    type LocalBasis<'a>: LocalBasis<T, X, N> where Self: 'a;

    /// Returns the [`Self::LocalBasis`] for the given `elem`.
    fn local_basis(&self, elem: &Self::Elem) -> Self::LocalBasis<'_>;
}