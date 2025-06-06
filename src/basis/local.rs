use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, RealField};
use crate::basis::traits::{Basis, NumBasis};

/// Set of basis functions with local support on each [element](Self::Elem).
///
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
pub trait LocalBasis<T: RealField, X>: NumBasis
    where DefaultAllocator: Allocator<<Self::ElemBasis as Basis<T, X>>::NumComponents, <Self::ElemBasis as Basis<T, X>>::NumBasis> 
{
    /// Element type.
    type Elem;
    
    /// Restriction of the local basis on an element.
    type ElemBasis: Basis<T, X>;

    // todo: possibly change to IntoIterator or separate trait/ struct all together
    /// Iterator over linear global indices.
    type GlobalIndices: Iterator<Item = usize> + Clone;

    /// Returns the [`Self::ElemBasis`] for the given `elem`,
    /// i.e. the restriction of this basis to the element.
    fn elem_basis(&self, elem: &Self::Elem) -> Self::ElemBasis;

    /// Returns an iterator over all global indices of the local basis of `elem`.
    fn global_indices(&self, local_basis: &Self::ElemBasis) -> Self::GlobalIndices;
}