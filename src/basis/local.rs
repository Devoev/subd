use crate::basis::traits::Basis;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dyn, RealField};

// todo: possibly remove Basis super trait again

/// Set of basis functions with local support on each [element](Self::Elem).
///
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
pub trait LocalBasis<T: RealField, X>: Basis<T, X, NumBasis = Dyn>
    where DefaultAllocator: Allocator<Self::NumComponents, <Self::ElemBasis as Basis<T, X>>::NumBasis>
{
    /// Element type.
    type Elem;
    
    /// Restriction of the local basis on an element.
    type ElemBasis: Basis<T, X, NumComponents = Self::NumComponents>;

    // todo: possibly change to IntoIterator or separate trait/ struct all together
    /// Iterator over linear global indices.
    type GlobalIndices: Iterator<Item = usize> + Clone;

    // todo: possibly change to Result<Self::Elem, ...>
    /// Finds the [`Self::Elem`] containing the given parametric value `x`.
    fn find_elem(&self, x: X) -> Self::Elem;

    /// Returns the [`Self::ElemBasis`] for the given `elem`,
    /// i.e. the restriction of this basis to the element.
    fn elem_basis(&self, elem: &Self::Elem) -> Self::ElemBasis;

    // todo: possibly change to take elem instead of local_basis
    /// Returns an iterator over all global indices of the local basis of `elem`.
    fn global_indices(&self, local_basis: &Self::ElemBasis) -> Self::GlobalIndices;
}