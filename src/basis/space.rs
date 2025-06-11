use crate::basis::traits::{Basis};
use std::marker::PhantomData;
use nalgebra::{DVector, DefaultAllocator, Dyn, OMatrix, RealField, U1};
use nalgebra::allocator::Allocator;
use crate::basis::eval::{EvalBasis, EvalGrad};
use crate::basis::lin_combination::LinCombination;
use crate::basis::local::LocalBasis;
use crate::index::dimensioned::Dimensioned;

/// Function space spanned by a set of basis functions of type [`B`]
/// as `V = span{b[1],...,b[n]}`.
///
/// See the [`Basis`] trait for information about type parameters.
#[derive(Debug, Clone, Copy)]
pub struct Space<T, X, B> {
    /// Set of basis functions spanning this function space.
    pub basis: B,

    _phantom_data: PhantomData<(T, X)>
}

impl <T, X, B: Basis> Space<T, X, B> {
    /// Constructs a new [`Space`] from the given `basis`.
    pub fn new(basis: B) -> Self {
        Self { basis, _phantom_data: PhantomData }
    }

    /// Returns the dimension of this space, i.e. the number of basis functions in its basis.
    pub fn dim(&self) -> usize {
        self.basis.num_basis()
    }
}

impl <T: RealField, X, B: Basis> Space<T, X, B> {
    /// Calculates the linear combination of the given `coeffs` with the basis function of this space,
    /// and returns the resulting [`LinCombination`].
    pub fn linear_combination(&self, coeffs: DVector<T>) -> LinCombination<T, X, B>  {
        LinCombination::new(coeffs, self)
    }
}

impl <T: RealField + Copy, X: Copy, B: LocalBasis<T, X>> Space<T, X, B>
    where DefaultAllocator: Allocator<B::NumComponents, <B::ElemBasis as Basis>::NumBasis>
{
    /// Evaluates only the local basis functions at the parametric point `x`.
    pub fn eval_local(&self, x: X) -> OMatrix<T, B::NumComponents, <B::ElemBasis as Basis>::NumBasis> {
        let elem = self.basis.find_elem(x);
        let local_basis = self.basis.elem_basis(&elem);
        local_basis.eval(x)
    }

    /// Evaluates only the local basis functions at the parametric point `x`.
    /// Returns the evaluated functions as well the indices corresponding to the global numbering.
    pub fn eval_local_with_idx(&self, x: X) -> (OMatrix<T, B::NumComponents, <B::ElemBasis as Basis>::NumBasis>, B::GlobalIndices) {
        let elem = self.basis.find_elem(x);
        let local_basis = self.basis.elem_basis(&elem);
        let b = local_basis.eval(x);
        let idx = self.basis.global_indices(&local_basis);
        (b, idx)
    }
}

// impl <T, X, BElem, B: LocalBasis<T, X, ElemBasis = BElem>, const D: usize> Space<T, X, B>
//     where T: RealField + Copy,
//           X: Dimensioned<T, D>,
//           BElem: EvalGrad<T, X, D>,
//           DefaultAllocator: Allocator<B::NumComponents, <B::ElemBasis as Basis>::NumBasis>, nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<nalgebra::Const<1>, <<B as basis::local::LocalBasis<T, X>>::ElemBasis as basis::traits::Basis>::NumBasis>
// {
//     /// Evaluates only the local basis functions at the parametric point `x`.
//     /// Returns the evaluated functions as well the indices corresponding to the global numbering.
//     pub fn eval_grad_local(&self, x: X) -> (OMatrix<T, B::NumComponents, <B::ElemBasis as Basis>::NumBasis>, B::GlobalIndices) {
//         let elem = self.basis.find_elem(x);
//         let local_basis = self.basis.elem_basis(&elem);
//         let b = local_basis.eval(x);
//         let idx = self.basis.global_indices(&local_basis);
//         (b, idx)
//     }
// }