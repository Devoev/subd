use std::ops::Mul;
use itertools::Itertools;
use crate::basis::space::Space;
use crate::basis::traits::{Basis, NumBasis};
use nalgebra::{ComplexField, Const, DVector, DefaultAllocator, Dyn, Matrix, OVector, RealField, SVector, VecStorage};
use nalgebra::allocator::Allocator;
use crate::basis::local::LocalBasis;

/// Linear combination of coefficients with basis functions.
pub struct LinCombination<'a, T: ComplexField, X, B> {
    /// Coefficients vector.
    pub coeffs: DVector<T>,

    /// Space of basis functions.
    pub space: &'a Space<T::RealField, X, B>,
}

impl <'a, T: ComplexField, X, B: NumBasis> LinCombination<'a, T, X, B> {
    /// Constructs a new [`LinCombination`] from the given `coeffs` and `space`.
    pub fn new(coeffs: DVector<T>, space: &'a Space<T::RealField, X, B>) -> Self {
        assert_eq!(coeffs.ncols(), space.dim(),
                   "The number of coefficients (is {}) must match the dimension of the space (is {})",
                   coeffs.ncols(), space.dim());

        LinCombination { coeffs, space }
    }
}

impl <'a, T, X, B> LinCombination<'a, T, X, B>
    where T: ComplexField, // todo: replace this with ComplexField and fix c * b in eval
          B: Basis<T::RealField, X, NumBasis=Dyn>,
          DefaultAllocator: Allocator<B::NumComponents, B::NumBasis>,
          DefaultAllocator: Allocator<B::NumComponents>,
{
    /// Evaluates this linear combination at the parametric point `x`,
    /// by calculating `c[0] * b[0](x) + ... + c[n] * b[n](x)`.
    pub fn eval(&self, x: X) -> OVector<T, B::NumComponents> {
        let b= self.space.basis.eval(x);
        // todo: is there another way to calculate this without using .map ?
        b.map(|bi| T::from_real(bi)) * &self.coeffs
    }
}

// impl <'a, T, X, B> LinCombination<'a, T, X, B>
//     where T: ComplexField, // todo: replace this with ComplexField and fix c * b in eval
//           B: LocalBasis<T::RealField, X>,
//           DefaultAllocator: Allocator<<B::ElemBasis as Basis<T, X>>::NumComponents>,
//           <B as LocalBasis<<T as ComplexField>::RealField, X>>::ElemBasis: Basis<T, X>
// {
//     // todo: this is ugly. change signature and how to get/provide the element
//     //  possibly add method to LocalBasis
//     /// Evaluates this linear combination at the parametric point `x`,
//     /// by calculating `c[0] * b[0](x) + ... + c[n] * b[n](x)`.
//     pub fn eval_on_elem(&self, elem: &B::Elem, x: X) -> OVector<T, <B::ElemBasis as Basis<T::RealField, X>>::NumComponents> {
//         let elem_basis = self.space.basis.elem_basis(elem);
//         let idx = self.space.basis.global_indices(&elem_basis);
//         let b = elem_basis.eval(x);
//         let c = self.coeffs.select_rows(idx.collect_vec().iter());
//         b * c
//     }
// }