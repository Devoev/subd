use std::iter::zip;
use crate::basis::eval::{EvalBasis, EvalGrad};
use crate::basis::lin_combination::LinCombination;
use crate::basis::local::LocalBasis;
use crate::basis::traits::Basis;
use crate::index::dimensioned::Dimensioned;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DVector, DefaultAllocator, MatrixView, MatrixViewMut, OMatrix, RealField, Vector, VectorView, U1};
use std::marker::PhantomData;
use std::ops::AddAssign;
use crate::basis::error::CoeffsSpaceDimError;

/// Function space spanned by a set of basis functions of type [`B`]
/// as `V = span{b[1],...,b[n]}`.
///
/// See the [`Basis`] trait for information about type parameters.
#[derive(Debug, Clone, Copy)]
pub struct Space<T, X, B, const D: usize> {
    /// Set of basis functions spanning this function space.
    pub basis: B,

    _phantom_data: PhantomData<(T, X)>
}

impl <T, X, B: Basis, const D: usize> Space<T, X, B, D> {
    /// Constructs a new [`Space`] from the given `basis`.
    pub fn new(basis: B) -> Self {
        Self { basis, _phantom_data: PhantomData }
    }

    /// Returns the dimension of this space, i.e. the number of basis functions in its basis.
    pub fn dim(&self) -> usize {
        self.basis.num_basis()
    }
}

impl <T: RealField, X, B: Basis, const D: usize> Space<T, X, B, D> {
    /// Calculates the linear combination of the given `coeffs` with the basis function of this space,
    /// and returns the resulting [`LinCombination`].
    pub fn linear_combination(&self, coeffs: DVector<T>) -> Result<LinCombination<T, X, B, D>, CoeffsSpaceDimError> {
        LinCombination::new(coeffs, self)
    }
}

/// Space of local basis functions of [`LocalBasis::elem_basis`].
type LocalSpace<T, X, B, const D: usize> = Space<T, X, <B as LocalBasis<T, X>>::ElemBasis, D>;

/// The local space [`LocalSpace`] paired with the indices [`LocalBasis::global_indices`]
/// of nonzero basis functions.
type LocalSpaceWithIdx<T, X, B, const D: usize> = (LocalSpace<T, X, B, D>, <B as LocalBasis<T, X>>::GlobalIndices);

impl <T: RealField + Copy, X, B: LocalBasis<T, X>, const D: usize> Space<T, X, B, D>
    where DefaultAllocator: Allocator<B::NumComponents, <B::ElemBasis as Basis>::NumBasis>
{
    /// Returns this space restricted to the local element `elem`.
    pub fn local_space(&self, elem: &B::Elem) -> LocalSpace<T, X, B, D> {
        Space::new(self.basis.elem_basis(elem))
    }

    /// Returns this space restricted to the local element `elem`
    /// as well the indices corresponding to the global numbering of local basis functions.
    pub fn local_space_with_idx(&self, elem: &B::Elem) -> LocalSpaceWithIdx<T, X, B, D> {
        (self.local_space(elem), self.basis.global_indices(elem))
    }
}

/// The owned matrix of [`Basis::NumComponents`] rows and [`Basis::NumBasis`] columns,
/// storing the evaluated basis functions of [`EvalBasis::eval`].
type EvalLocal<T, X, B> = OMatrix<T, <B as Basis>::NumComponents, <<B as LocalBasis<T, X>>::ElemBasis as Basis>::NumBasis>;

/// The basis matrix [`EvalLocal`] paired with the indices [`LocalBasis::global_indices`]
/// of nonzero basis functions.
type EvalLocalWithIdx<T, X, B> = (EvalLocal<T, X, B>, <B as LocalBasis<T, X>>::GlobalIndices);

impl <T: RealField, X: Copy, B: LocalBasis<T, X>, const D: usize> Space<T, X, B, D>
    where DefaultAllocator: Allocator<B::NumComponents, <B::ElemBasis as Basis>::NumBasis>
{
    /// Evaluates only the local basis functions at the parametric point `x`.
    pub fn eval_local(&self, x: X) -> EvalLocal<T, X, B> {
        self.eval_on_elem(&self.basis.find_elem(x), x)
    }

    /// Evaluates only the local basis functions on the given `elem` at the parametric point `x`.
    pub fn eval_on_elem(&self, elem: &B::Elem, x: X) -> EvalLocal<T, X, B> {
        let local_basis = self.basis.elem_basis(elem);
        local_basis.eval(x)
    }

    /// Evaluates only the local basis functions at the parametric point `x`.
    /// Returns the evaluated functions as well the indices corresponding to the global numbering.
    pub fn eval_local_with_idx(&self, x: X) -> EvalLocalWithIdx<T, X, B> {
        self.eval_on_elem_with_idx(&self.basis.find_elem(x), x)
    }

    /// Evaluates only the local basis functions on the given `elem` at the parametric point `x`.
    /// Returns the evaluated functions as well the indices corresponding to the global numbering.
    pub fn eval_on_elem_with_idx(&self, elem: &B::Elem, x: X) -> EvalLocalWithIdx<T, X, B> {
        let local_basis = self.basis.elem_basis(elem);
        let b = local_basis.eval(x);
        let idx = self.basis.global_indices(elem);
        (b, idx)
    }

    /// Populates the global basis matrix `global` with the local basis values evaluated at `x`.
    pub fn populate_global(&self, global: &mut OMatrix<T, B::NumComponents, B::NumBasis>, x: X) {
        let (b, idx) = self.eval_local_with_idx(x);
        for (i_global, bi) in zip(idx, b.column_iter()) {
            global.column_mut(i_global).add_assign(bi) // todo: should this be replace with copy_from?
        }
    }
}

/// The owned matrix of [`D`] rows and [`Basis::NumBasis`] columns,
/// storing the gradients of basis functions of [EvalGrad::eval_grad].
type EvalGradLocal<T, X, B, const D: usize> = OMatrix<T, Const<D>, <<B as LocalBasis<T, X>>::ElemBasis as Basis>::NumBasis>;

/// The gradient matrix [`EvalGradLocal`] paired with the indices [`LocalBasis::global_indices`]
/// of nonzero basis functions.
type EvalGradLocalWithIdx<T, X, B, const D: usize> = (EvalGradLocal<T, X, B, D>, <B as LocalBasis<T, X>>::GlobalIndices);

impl <T, X, B, const D: usize> Space<T, X, B, D>
    where T: RealField + Copy,
          X: Dimensioned<T, D> + Copy,
          B: LocalBasis<T, X>,
          B::ElemBasis: EvalGrad<T, X, D>,
          DefaultAllocator: Allocator<B::NumComponents, <B::ElemBasis as Basis>::NumBasis>,
          DefaultAllocator: Allocator<U1, <B::ElemBasis as Basis>::NumBasis>,
          DefaultAllocator: Allocator<Const<D>, <B::ElemBasis as Basis>::NumBasis>,
{
    /// Evaluates the gradients of only local basis functions at the parametric point `x`.
    pub fn eval_grad_local(&self, x: X) -> EvalGradLocal<T, X, B, D> {
        let elem = self.basis.find_elem(x);
        let local_basis = self.basis.elem_basis(&elem);
        local_basis.eval_grad(x)
    }

    /// valuates the gradients of only local basis functions at the parametric point `x`.
    /// Returns the evaluated gradients as well the indices corresponding to the global numbering.
    pub fn eval_grad_local_with_idx(&self, x: X) -> EvalGradLocalWithIdx<T, X, B, D> {
        let elem = self.basis.find_elem(x);
        let local_basis = self.basis.elem_basis(&elem);
        let b = local_basis.eval_grad(x);
        let idx = self.basis.global_indices(&elem);
        (b, idx)
    }

    /// Populates the global basis gradient matrix `global`
    /// with the local basis gradient values evaluated at `x`.
    pub fn populate_grad_global(&self, global: &mut OMatrix<T, Const<D>, B::NumBasis>, x: X) {
        let (b, idx) = self.eval_grad_local_with_idx(x);
        for (i_global, bi) in zip(idx, b.column_iter()) {
            global.column_mut(i_global).add_assign(bi) // todo: should this be replace with copy_from?
        }
    }
}