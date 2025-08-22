use crate::basis::eval::{EvalBasis, EvalBasisAllocator, EvalGrad, EvalGradAllocator};
use crate::basis::traits::Basis;
use nalgebra::{DefaultAllocator, Dyn, RealField, Scalar, U1};

// todo: NumBasis from basis super-trait is never used. Can this be removed?

/// Set of basis functions with local support on each [element](Self::Elem).
///
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
pub trait LocalBasis<T: Scalar>: Basis<NumBasis = Dyn>
    where DefaultAllocator: EvalBasisAllocator<Self::ElemBasis>
{
    /// Element type.
    type Elem;
    
    /// Restriction of the local basis on an element.
    type ElemBasis: EvalBasis<T, NumComponents = Self::NumComponents, Coord<T> = Self::Coord<T>>;

    // todo: possibly change to IntoIterator or separate trait/ struct all together
    /// Iterator over linear global indices.
    type GlobalIndices: Iterator<Item = usize> + Clone;

    /// Returns the [`Self::ElemBasis`] for the given `elem`,
    /// i.e. the restriction of this basis to the element.
    fn elem_basis(&self, elem: &Self::Elem) -> Self::ElemBasis;

    /// Returns an iterator over all global indices of the local basis of `elem`.
    fn global_indices(&self, elem: &Self::Elem) -> Self::GlobalIndices;
}

/// Local basis functions with gradient evaluations.
pub trait LocalGradBasis<T: RealField, const D: usize>: LocalBasis<T, ElemBasis: EvalGrad<T, D>, NumComponents = U1>
    where DefaultAllocator: EvalGradAllocator<Self::ElemBasis, D> {}

impl <T: RealField, const D: usize, B> LocalGradBasis<T, D> for B
where B: LocalBasis<T, ElemBasis: EvalGrad<T, D>, NumComponents = U1>, 
      DefaultAllocator: EvalGradAllocator<Self::ElemBasis, D>
{}

/// Local basis functions that can find the local element by parametric value.
pub trait FindElem<T: Scalar>: LocalBasis<T>
    where DefaultAllocator: EvalBasisAllocator<Self::ElemBasis>
{
    // todo: possibly change to Result<Self::Elem, ...>
    /// Finds the [`Self::Elem`] containing the given parametric value `x`.
    fn find_elem(&self, x: Self::Coord<T>) -> Self::Elem;
}