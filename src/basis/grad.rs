use nalgebra::{ComplexField, Const, DefaultAllocator, OMatrix, RealField, U1};
use nalgebra::allocator::Allocator;
use crate::basis::eval::{EvalBasis, EvalGrad};
use crate::basis::lin_combination::LinCombination;
use crate::basis::local::LocalBasis;
use crate::basis::space::Space;
use crate::basis::traits::Basis;

/// Gradient of basis functions `grad B = { grad b[i] : b[i] ∈ B }`.
pub struct GradBasis<B, const D: usize>(B);

impl<B: Basis<NumComponents = U1>, const D: usize> Basis for GradBasis<B, D> {
    type NumBasis = B::NumBasis;
    type NumComponents = Const<D>;

    fn num_basis_generic(&self) -> Self::NumBasis {
        self.0.num_basis_generic()
    }
}

/// Implement [`EvalBasis`] if `B` implements [`EvalBasis`].
impl <T: RealField, X, B: EvalGrad<T, X, D>, const D: usize> EvalBasis<T, X> for GradBasis<B, D>
where DefaultAllocator: Allocator<U1, Self::NumBasis>,
      DefaultAllocator: Allocator<Const<D>, Self::NumBasis>
{
    fn eval(&self, x: X) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        self.0.eval_grad(x)
    }
}

/// Implement [`LocalBasis`] if `B` is also a local basis.
impl <T, X, B, const D: usize> LocalBasis<T, X> for GradBasis<B, D>
where T: RealField,
      B: LocalBasis<T, X, NumComponents = U1>,
      B::ElemBasis: EvalGrad<T, X, D>,
      DefaultAllocator: Allocator<B::NumComponents, <B::ElemBasis as Basis>::NumBasis>,
      DefaultAllocator: Allocator<U1, <B::ElemBasis as Basis>::NumBasis>,
      DefaultAllocator: Allocator<Const<D>, <B::ElemBasis as Basis>::NumBasis>,
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
pub type GradSpace<T, X, B, const D: usize> = Space<T, X, GradBasis<B, D>, D>;

impl <T, X, B, const D: usize> Space<T, X, B, D>
where T: RealField,
      B: LocalBasis<T, X, NumComponents = U1>,
      B::ElemBasis: EvalGrad<T, X, D>,
      DefaultAllocator: Allocator<B::NumComponents, <B::ElemBasis as Basis>::NumBasis>,
      DefaultAllocator: Allocator<U1, <B::ElemBasis as Basis>::NumBasis>,
      DefaultAllocator: Allocator<Const<D>, <B::ElemBasis as Basis>::NumBasis>,
{
    /// Returns the gradient of this space.
    pub fn grad(self) -> GradSpace<T, X, B, D> {
        let basis = self.basis;
        Space::new(GradBasis(basis))
    }
}

impl <'a, T, X, B, const D: usize> LinCombination<'a, T, X, B, D>
    where T: ComplexField,
          X: Copy,
          B: LocalBasis<T::RealField, X, NumComponents = U1>,
          B::ElemBasis: EvalGrad<T::RealField, X, D>,
          DefaultAllocator: Allocator<U1>,
          DefaultAllocator: Allocator<<B::ElemBasis as Basis>::NumBasis>,
          DefaultAllocator: Allocator<U1, <B::ElemBasis as Basis>::NumBasis>,
          DefaultAllocator: Allocator<Const<D>, <B::ElemBasis as Basis>::NumBasis>
{
    /// Returns the gradient of this linear combination in the space `grad_space`.
    pub fn grad(self, grad_space: &'a GradSpace<T::RealField, X, B, D>) -> LinCombination<'a, T, X, GradBasis<B, D>, D> {
        LinCombination::new(self.coeffs, grad_space).unwrap()
    }
}