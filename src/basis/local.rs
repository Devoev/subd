use crate::basis::traits::Basis;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dyn, RealField};
use crate::basis::eval::{EvalBasis, EvalBasisAllocator};

// todo: NumBasis from basis super-trait is never used. Can this be removed?

/// Set of basis functions with local support on each [element](Self::Elem).
///
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
pub trait LocalBasis<T: RealField, X>: Basis<NumBasis = Dyn>
    where DefaultAllocator: EvalBasisAllocator<Self::ElemBasis>
{
    /// Element type.
    type Elem;
    
    /// Restriction of the local basis on an element.
    type ElemBasis: EvalBasis<T, X, NumComponents = Self::NumComponents>;

    // todo: possibly change to IntoIterator or separate trait/ struct all together
    /// Iterator over linear global indices.
    type GlobalIndices: Iterator<Item = usize> + Clone;

    /// Returns the [`Self::ElemBasis`] for the given `elem`,
    /// i.e. the restriction of this basis to the element.
    fn elem_basis(&self, elem: &Self::Elem) -> Self::ElemBasis;

    /// Returns an iterator over all global indices of the local basis of `elem`.
    fn global_indices(&self, elem: &Self::Elem) -> Self::GlobalIndices;
}

/// Local basis functions that can find the local element by parametric value.
pub trait FindElem<T: RealField, X>: LocalBasis<T, X>
    where DefaultAllocator: EvalBasisAllocator<Self::ElemBasis>
{
    // todo: possibly change to Result<Self::Elem, ...>
    /// Finds the [`Self::Elem`] containing the given parametric value `x`.
    fn find_elem(&self, x: X) -> Self::Elem;
}