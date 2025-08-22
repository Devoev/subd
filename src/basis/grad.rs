use crate::basis::eval::{EvalBasis, EvalGrad, EvalGradAllocator};
use crate::basis::lin_combination::LinCombination;
use crate::basis::local::{LocalBasis, LocalGradBasis};
use crate::basis::space::Space;
use crate::basis::traits::Basis;
use nalgebra::{ComplexField, Const, DefaultAllocator, OMatrix, RealField, U1};

/// Gradient of basis functions `grad B = { grad b[i] : b[i] ∈ B }`.
pub struct GradBasis<B, const D: usize>(B);

impl<B: Basis<NumComponents = U1>, const D: usize> Basis for GradBasis<B, D> {
    type NumBasis = B::NumBasis;
    type NumComponents = Const<D>;
    type Coord<T> = B::Coord<T>;

    fn num_basis_generic(&self) -> Self::NumBasis {
        self.0.num_basis_generic()
    }
}

/// Implement [`EvalBasis`] if `B` implements [`EvalBasis`].
impl <T: RealField, B: EvalGrad<T, D>, const D: usize> EvalBasis<T> for GradBasis<B, D>
    where DefaultAllocator: EvalGradAllocator<B, D>
{
    fn eval(&self, x: Self::Coord<T>) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        self.0.eval_grad(x)
    }
}

/// Implement [`LocalBasis`] if `B` is also a local basis.
impl <T, B, const D: usize> LocalBasis<T> for GradBasis<B, D>
where T: RealField,
      B: LocalGradBasis<T, D>,
      DefaultAllocator: EvalGradAllocator<B::ElemBasis, D>
{
    type Elem = B::Elem;
    type ElemBasis = GradBasis<B::ElemBasis, D>;
    type GlobalIndices = B::GlobalIndices;

    fn elem_basis(&self, elem: &Self::Elem) -> Self::ElemBasis {
        GradBasis(self.0.elem_basis(elem))
    }

    fn global_indices(&self, elem: &Self::Elem) -> Self::GlobalIndices {
        self.0.global_indices(elem)
    }
}

/// Space of gradients of basis functions in `B`.
pub type GradSpace<T, B, const D: usize> = Space<T, GradBasis<B, D>, D>;

impl <T, B, const D: usize> Space<T, B, D>
where T: RealField,
      B: LocalGradBasis<T, D>,
      DefaultAllocator: EvalGradAllocator<B::ElemBasis, D>
{
    /// Returns the gradient of this space.
    pub fn grad(self) -> GradSpace<T, B, D> {
        let basis = self.basis;
        Space::new(GradBasis(basis))
    }
}

impl <'a, T, B, const D: usize> LinCombination<'a, T, B, D>
    where T: ComplexField,
          B: LocalGradBasis<T::RealField, D>,
          DefaultAllocator: EvalGradAllocator<B::ElemBasis, D>
{
    /// Returns the gradient of this linear combination in the space `grad_space`.
    pub fn grad(self, grad_space: &'a GradSpace<T::RealField, B, D>) -> LinCombination<'a, T, GradBasis<B, D>, D> {
        LinCombination::new(self.coeffs, grad_space).unwrap()
    }
}