use crate::basis::error::CoeffsSpaceDimError;
use crate::basis::eval::{EvalBasis, EvalBasisAllocator, EvalGrad, EvalGradAllocator};
use crate::basis::lin_combination::LinCombination;
use crate::basis::local::{FindElem, MeshBasis, MeshGradBasis};
use crate::basis::traits::Basis;
use nalgebra::{Const, DVector, DefaultAllocator, OMatrix, RealField};
use std::iter::zip;
use std::marker::PhantomData;
use std::ops::AddAssign;

/// Function space spanned by a set of basis functions of type [`B`]
/// as `V = span{b[1],...,b[n]}`.
///
/// See the [`Basis`] trait for information about type parameters.
#[derive(Debug, Clone, Copy)]
pub struct Space<T, B, const D: usize> {
    /// Set of basis functions spanning this function space.
    pub basis: B,

    _phantom_data: PhantomData<T>
}

impl <T, B: Basis, const D: usize> Space<T, B, D> {
    /// Constructs a new [`Space`] from the given `basis`.
    pub fn new(basis: B) -> Self {
        Self { basis, _phantom_data: PhantomData }
    }

    /// Returns the dimension of this space, i.e. the number of basis functions in its basis.
    pub fn dim(&self) -> usize {
        self.basis.num_basis()
    }
}

impl <T: RealField, B: Basis, const D: usize> Space<T, B, D> {
    /// Calculates the linear combination of the given `coeffs` with the basis function of this space,
    /// and returns the resulting [`LinCombination`].
    pub fn linear_combination(&self, coeffs: DVector<T>) -> Result<LinCombination<T, B, D>, CoeffsSpaceDimError> {
        LinCombination::new(coeffs, self)
    }
}

/// Space of local basis functions of [`MeshBasis::local_basis`].
type LocalSpace<T, B, const D: usize> = Space<T, <B as MeshBasis<T>>::LocalBasis, D>;

/// The local space [`LocalSpace`] paired with the indices [`MeshBasis::global_indices`]
/// of nonzero basis functions.
type LocalSpaceWithIdx<T, B, const D: usize> = (LocalSpace<T, B, D>, <B as MeshBasis<T>>::GlobalIndices);

impl <T: RealField, B: MeshBasis<T>, const D: usize> Space<T, B, D>
    where DefaultAllocator: EvalBasisAllocator<B::LocalBasis>
{
    /// Returns this space restricted to the local element `elem`.
    pub fn local_space(&self, elem: &B::Cell) -> LocalSpace<T, B, D> {
        Space::new(self.basis.local_basis(elem))
    }

    /// Returns this space restricted to the local element `elem`
    /// as well the indices corresponding to the global numbering of local basis functions.
    pub fn local_space_with_idx(&self, elem: &B::Cell) -> LocalSpaceWithIdx<T, B, D> {
        (self.local_space(elem), self.basis.global_indices(elem))
    }
}

/// The owned matrix of [`Basis::NumComponents`] rows and [`Basis::NumBasis`] columns,
/// storing the evaluated basis functions of [`EvalBasis::eval`].
type EvalLocal<T, B> = OMatrix<T, <B as Basis>::NumComponents, <<B as MeshBasis<T>>::LocalBasis as Basis>::NumBasis>;

/// The basis matrix [`EvalLocal`] paired with the indices [`MeshBasis::global_indices`]
/// of nonzero basis functions.
type EvalLocalWithIdx<T, B> = (EvalLocal<T, B>, <B as MeshBasis<T>>::GlobalIndices);

impl <T: RealField, B: MeshBasis<T>, const D: usize> Space<T, B, D>
    where DefaultAllocator: EvalBasisAllocator<B::LocalBasis>
{
    /// Evaluates only the local basis functions on the given `elem` at the parametric point `x`.
    pub fn eval_on_elem(&self, elem: &B::Cell, x: B::Coord<T>) -> EvalLocal<T, B> {
        let local_basis = self.basis.local_basis(elem);
        local_basis.eval(x)
    }

    /// Evaluates only the local basis functions on the given `elem` at the parametric point `x`.
    /// Returns the evaluated functions as well the indices corresponding to the global numbering.
    pub fn eval_on_elem_with_idx(&self, elem: &B::Cell, x: B::Coord<T>) -> EvalLocalWithIdx<T, B> {
        let local_basis = self.basis.local_basis(elem);
        let b = local_basis.eval(x);
        let idx = self.basis.global_indices(elem);
        (b, idx)
    }

    /// Populates the global basis matrix `global` 
    /// with the local basis values on `elem` evaluated at `x`.
    pub fn populate_global_on_elem(&self, global: &mut OMatrix<T, B::NumComponents, B::NumBasis>, elem: &B::Cell, x: B::Coord<T>) {
        let (b, idx) = self.eval_on_elem_with_idx(elem, x);
        for (i_global, bi) in zip(idx, b.column_iter()) {
            global.column_mut(i_global).add_assign(bi) // todo: should this be replace with copy_from?
        }
    }
}

impl <T: RealField, B: FindElem<T>, const D: usize> Space<T, B, D>
    where B::Coord<T>: Copy,
          DefaultAllocator: EvalBasisAllocator<B::LocalBasis>
{
    /// Evaluates only the local basis functions at the parametric point `x`.
    pub fn eval_local(&self, x: B::Coord<T>) -> EvalLocal<T, B> {
        self.eval_on_elem(&self.basis.find_elem(x), x)
    }

    /// Evaluates only the local basis functions at the parametric point `x`.
    /// Returns the evaluated functions as well the indices corresponding to the global numbering.
    pub fn eval_local_with_idx(&self, x: B::Coord<T>) -> EvalLocalWithIdx<T, B> {
        self.eval_on_elem_with_idx(&self.basis.find_elem(x), x)
    }

    /// Populates the global basis matrix `global` with the local basis values evaluated at `x`.
    pub fn populate_global(&self, global: &mut OMatrix<T, B::NumComponents, B::NumBasis>, x: B::Coord<T>) {
        let (b, idx) = self.eval_local_with_idx(x);
        for (i_global, bi) in zip(idx, b.column_iter()) {
            global.column_mut(i_global).add_assign(bi) // todo: should this be replace with copy_from?
        }
    }
}

/// The owned matrix of [`D`] rows and [`Basis::NumBasis`] columns,
/// storing the gradients of basis functions of [EvalGrad::eval_grad].
type EvalGradLocal<T, B, const D: usize> = OMatrix<T, Const<D>, <<B as MeshBasis<T>>::LocalBasis as Basis>::NumBasis>;

/// The gradient matrix [`EvalGradLocal`] paired with the indices [`MeshBasis::global_indices`]
/// of nonzero basis functions.
type EvalGradLocalWithIdx<T, B, const D: usize> = (EvalGradLocal<T, B, D>, <B as MeshBasis<T>>::GlobalIndices);

impl <T, B, const D: usize> Space<T, B, D>
    where T: RealField,
          B: MeshGradBasis<T, D>,
          DefaultAllocator: EvalGradAllocator<B::LocalBasis, D>
{
    /// Evaluates the gradients of only local basis functions on the given `elem` at the parametric point `x`.
    pub fn eval_grad_on_elem(&self, elem: &B::Cell, x: B::Coord<T>) -> EvalGradLocal<T, B, D> {
        let local_basis = self.basis.local_basis(elem);
        local_basis.eval_grad(x)
    }

    /// valuates the gradients of only local basis functions at the parametric point `x`.
    /// Returns the evaluated gradients as well the indices corresponding to the global numbering.
    pub fn eval_grad_on_elem_with_idx(&self, elem: &B::Cell, x: B::Coord<T>) -> EvalGradLocalWithIdx<T, B, D> {
        let local_basis = self.basis.local_basis(elem);
        let b = local_basis.eval_grad(x);
        let idx = self.basis.global_indices(elem);
        (b, idx)
    }

    /// Populates the global basis gradient matrix `global`
    /// with the local basis gradient values evaluated at `x`.
    pub fn populate_grad_global_on_elem(&self, global: &mut OMatrix<T, Const<D>, B::NumBasis>, elem: &B::Cell, x: B::Coord<T>) {
        let (b, idx) = self.eval_grad_on_elem_with_idx(elem, x);
        for (i_global, bi) in zip(idx, b.column_iter()) {
            global.column_mut(i_global).add_assign(bi) // todo: should this be replace with copy_from?
        }
    }
}

impl <T, B, const D: usize> Space<T, B, D>
    where T: RealField,
          B: FindElem<T> + MeshGradBasis<T, D>,
          B::Coord<T>: Copy,
          DefaultAllocator: EvalGradAllocator<B::LocalBasis, D>
{
    /// Evaluates the gradients of only local basis functions at the parametric point `x`.
    pub fn eval_grad_local(&self, x: B::Coord<T>) -> EvalGradLocal<T, B, D> {
        self.eval_grad_on_elem(&self.basis.find_elem(x), x)
    }

    /// valuates the gradients of only local basis functions at the parametric point `x`.
    /// Returns the evaluated gradients as well the indices corresponding to the global numbering.
    pub fn eval_grad_local_with_idx(&self, x: B::Coord<T>) -> EvalGradLocalWithIdx<T, B, D> {
        self.eval_grad_on_elem_with_idx(&self.basis.find_elem(x), x)
    }

    /// Populates the global basis gradient matrix `global`
    /// with the local basis gradient values evaluated at `x`.
    pub fn populate_grad_global(&self, global: &mut OMatrix<T, Const<D>, B::NumBasis>, x: B::Coord<T>) {
        let (b, idx) = self.eval_grad_local_with_idx(x);
        for (i_global, bi) in zip(idx, b.column_iter()) {
            global.column_mut(i_global).add_assign(bi) // todo: should this be replace with copy_from?
        }
    }
}