use crate::space::error::CoeffsSpaceDimError;
use crate::space::eval_basis::{EvalBasis, EvalBasisAllocator, EvalGrad, EvalGradAllocator};
use crate::space::lin_combination::LinCombination;
use crate::space::local::{FindElem, MeshBasis, MeshGradBasis};
use crate::space::basis::BasisFunctions;
use nalgebra::{Const, DVector, DefaultAllocator, OMatrix, RealField};
use std::iter::zip;
use std::marker::PhantomData;
use std::ops::AddAssign;

pub mod tensor_prod;
pub mod local;
pub mod basis;
pub mod lin_combination;
pub mod eval_basis;
pub mod error;
pub mod cart_prod;
pub mod grad;

/// The most generic function space type.
///
/// A function space is spanned by a set of `Basis` functions
/// as `V = span{b[1],...,b[n]}`.
///
/// See the [`BasisFunctions`] trait for information about basis functions.
#[derive(Debug, Clone, Copy)]
pub struct Space<T, Basis> {
    /// Set of basis functions spanning this function space.
    pub basis: Basis,

    _phantom_data: PhantomData<T>
}

impl <T, Basis: BasisFunctions> Space<T, Basis> {
    /// Constructs a new `Space` from the given `basis`.
    pub fn new(basis: Basis) -> Self {
        Self { basis, _phantom_data: PhantomData }
    }

    /// Returns the dimension of this space, i.e. the number of basis functions in its basis.
    pub fn dim(&self) -> usize {
        self.basis.num_basis()
    }
}

impl <T: RealField, Basis: BasisFunctions> Space<T, Basis> {
    /// Calculates the linear combination of the given `coeffs` with the basis function of this space,
    /// and returns the resulting [`LinCombination`].
    pub fn linear_combination(&self, coeffs: DVector<T>) -> Result<LinCombination<T, Basis>, CoeffsSpaceDimError> {
        LinCombination::new(coeffs, self)
    }
}

/// Space of local basis functions of [`MeshBasis::local_basis`].
type LocalSpace<T, Basis> = Space<T, <Basis as MeshBasis<T>>::LocalBasis>;

/// The local space [`LocalSpace`] paired with the indices [`MeshBasis::global_indices`]
/// of nonzero basis functions.
type LocalSpaceWithIdx<T, Basis> = (LocalSpace<T, Basis>, <Basis as MeshBasis<T>>::GlobalIndices);

impl <T: RealField, Basis: MeshBasis<T>> Space<T, Basis>
where DefaultAllocator: EvalBasisAllocator<Basis::LocalBasis>
{
    /// Returns this space restricted to the local element `elem`.
    pub fn local_space(&self, elem: &Basis::Cell) -> LocalSpace<T, Basis> {
        Space::new(self.basis.local_basis(elem))
    }

    /// Returns this space restricted to the local element `elem`
    /// as well the indices corresponding to the global numbering of local basis functions.
    pub fn local_space_with_idx(&self, elem: &Basis::Cell) -> LocalSpaceWithIdx<T, Basis> {
        (self.local_space(elem), self.basis.global_indices(elem))
    }
}

/// The owned matrix of [`BasisFunctions::NumComponents`] rows and [`BasisFunctions::NumBasis`] columns,
/// storing the evaluated basis functions of [`EvalBasis::eval`].
type EvalLocal<T, Basis> = OMatrix<T, <Basis as BasisFunctions>::NumComponents, <<Basis as MeshBasis<T>>::LocalBasis as BasisFunctions>::NumBasis>;

/// The basis matrix [`EvalLocal`] paired with the indices [`MeshBasis::global_indices`]
/// of nonzero basis functions.
type EvalLocalWithIdx<T, Basis> = (EvalLocal<T, Basis>, <Basis as MeshBasis<T>>::GlobalIndices);

impl <T: RealField, Basis: MeshBasis<T>> Space<T, Basis>
where DefaultAllocator: EvalBasisAllocator<Basis::LocalBasis>
{
    /// Evaluates only the local basis functions on the given `elem` at the parametric point `x`.
    pub fn eval_on_elem(&self, elem: &Basis::Cell, x: Basis::Coord<T>) -> EvalLocal<T, Basis> {
        let local_basis = self.basis.local_basis(elem);
        local_basis.eval(x)
    }

    /// Evaluates only the local basis functions on the given `elem` at the parametric point `x`.
    /// Returns the evaluated functions as well the indices corresponding to the global numbering.
    pub fn eval_on_elem_with_idx(&self, elem: &Basis::Cell, x: Basis::Coord<T>) -> EvalLocalWithIdx<T, Basis> {
        let local_basis = self.basis.local_basis(elem);
        let b = local_basis.eval(x);
        let idx = self.basis.global_indices(elem);
        (b, idx)
    }

    /// Populates the global basis matrix `global`
    /// with the local basis values on `elem` evaluated at `x`.
    pub fn populate_global_on_elem(&self, global: &mut OMatrix<T, Basis::NumComponents, Basis::NumBasis>, elem: &Basis::Cell, x: Basis::Coord<T>) {
        let (b, idx) = self.eval_on_elem_with_idx(elem, x);
        for (i_global, bi) in zip(idx, b.column_iter()) {
            global.column_mut(i_global).add_assign(bi) // todo: should this be replace with copy_from?
        }
    }
}

impl <T: RealField, Basis: FindElem<T>> Space<T, Basis>
where Basis::Coord<T>: Copy,
      DefaultAllocator: EvalBasisAllocator<Basis::LocalBasis>
{
    /// Evaluates only the local basis functions at the parametric point `x`.
    pub fn eval_local(&self, x: Basis::Coord<T>) -> EvalLocal<T, Basis> {
        self.eval_on_elem(&self.basis.find_elem(x), x)
    }

    /// Evaluates only the local basis functions at the parametric point `x`.
    /// Returns the evaluated functions as well the indices corresponding to the global numbering.
    pub fn eval_local_with_idx(&self, x: Basis::Coord<T>) -> EvalLocalWithIdx<T, Basis> {
        self.eval_on_elem_with_idx(&self.basis.find_elem(x), x)
    }

    /// Populates the global basis matrix `global` with the local basis values evaluated at `x`.
    pub fn populate_global(&self, global: &mut OMatrix<T, Basis::NumComponents, Basis::NumBasis>, x: Basis::Coord<T>) {
        let (b, idx) = self.eval_local_with_idx(x);
        for (i_global, bi) in zip(idx, b.column_iter()) {
            global.column_mut(i_global).add_assign(bi) // todo: should this be replace with copy_from?
        }
    }
}

/// The owned matrix of [`D`] rows and [`BasisFunctions::NumBasis`] columns,
/// storing the gradients of basis functions of [EvalGrad::eval_grad].
type EvalGradLocal<T, Basis> = OMatrix<T, <Basis as BasisFunctions>::ParametricDim, <<Basis as MeshBasis<T>>::LocalBasis as BasisFunctions>::NumBasis>;

/// The gradient matrix [`EvalGradLocal`] paired with the indices [`MeshBasis::global_indices`]
/// of nonzero basis functions.
type EvalGradLocalWithIdx<T, Basis> = (EvalGradLocal<T, Basis>, <Basis as MeshBasis<T>>::GlobalIndices);

impl <T, Basis> Space<T, Basis>
where T: RealField,
      Basis: MeshGradBasis<T>,
      DefaultAllocator: EvalGradAllocator<Basis::LocalBasis>
{
    /// Evaluates the gradients of only local basis functions on the given `elem` at the parametric point `x`.
    pub fn eval_grad_on_elem(&self, elem: &Basis::Cell, x: Basis::Coord<T>) -> EvalGradLocal<T, Basis> {
        let local_basis = self.basis.local_basis(elem);
        local_basis.eval_grad(x)
    }

    /// valuates the gradients of only local basis functions at the parametric point `x`.
    /// Returns the evaluated gradients as well the indices corresponding to the global numbering.
    pub fn eval_grad_on_elem_with_idx(&self, elem: &Basis::Cell, x: Basis::Coord<T>) -> EvalGradLocalWithIdx<T, Basis> {
        let local_basis = self.basis.local_basis(elem);
        let b = local_basis.eval_grad(x);
        let idx = self.basis.global_indices(elem);
        (b, idx)
    }

    /// Populates the global basis gradient matrix `global`
    /// with the local basis gradient values evaluated at `x`.
    pub fn populate_grad_global_on_elem(&self, global: &mut OMatrix<T, Basis::ParametricDim, Basis::NumBasis>, elem: &Basis::Cell, x: Basis::Coord<T>) {
        let (b, idx) = self.eval_grad_on_elem_with_idx(elem, x);
        for (i_global, bi) in zip(idx, b.column_iter()) {
            global.column_mut(i_global).add_assign(bi) // todo: should this be replace with copy_from?
        }
    }
}

impl <T, Basis> Space<T, Basis>
where T: RealField,
      Basis: FindElem<T> + MeshGradBasis<T>,
      Basis::Coord<T>: Copy,
      DefaultAllocator: EvalGradAllocator<Basis::LocalBasis>
{
    /// Evaluates the gradients of only local basis functions at the parametric point `x`.
    pub fn eval_grad_local(&self, x: Basis::Coord<T>) -> EvalGradLocal<T, Basis> {
        self.eval_grad_on_elem(&self.basis.find_elem(x), x)
    }

    /// valuates the gradients of only local basis functions at the parametric point `x`.
    /// Returns the evaluated gradients as well the indices corresponding to the global numbering.
    pub fn eval_grad_local_with_idx(&self, x: Basis::Coord<T>) -> EvalGradLocalWithIdx<T, Basis> {
        self.eval_grad_on_elem_with_idx(&self.basis.find_elem(x), x)
    }

    /// Populates the global basis gradient matrix `global`
    /// with the local basis gradient values evaluated at `x`.
    pub fn populate_grad_global(&self, global: &mut OMatrix<T, Basis::ParametricDim, Basis::NumBasis>, x: Basis::Coord<T>) {
        let (b, idx) = self.eval_grad_local_with_idx(x);
        for (i_global, bi) in zip(idx, b.column_iter()) {
            global.column_mut(i_global).add_assign(bi) // todo: should this be replace with copy_from?
        }
    }
}