use crate::basis::error::CoeffsSpaceDimError;
use crate::basis::eval::EvalBasis;
use crate::basis::space::Space;
use crate::basis::traits::Basis;
use nalgebra::allocator::Allocator;
use nalgebra::{ComplexField, DVector, DefaultAllocator, Dyn, OVector};

/// Linear combination of coefficients with basis functions.
pub struct LinCombination<'a, T: ComplexField, X, B, const D: usize> {
    /// Coefficients vector.
    pub coeffs: DVector<T>,

    /// Space of basis functions.
    pub space: &'a Space<T::RealField, X, B, D>,
}

impl <'a, T: ComplexField, X, B: Basis, const D: usize> LinCombination<'a, T, X, B, D> {
    /// Constructs a new [`LinCombination`] from the given `coeffs` and `space`.
    /// 
    /// # Errors
    /// Will return an error if the number of rows of `coeffs`
    /// does not match the dimension of `space`.
    pub fn new(coeffs: DVector<T>, space: &'a Space<T::RealField, X, B, D>) -> Result<Self, CoeffsSpaceDimError> {
        match coeffs.nrows() == space.dim() {
            true => Ok(LinCombination { coeffs, space }),
            false => Err(CoeffsSpaceDimError { num_coeffs: coeffs.nrows(), dim_space: space.dim() })
        }
    }
}

impl <'a, T, X, B, const D: usize> LinCombination<'a, T, X, B, D>
    where T: ComplexField, // todo: replace this with ComplexField and fix c * b in eval
          B: EvalBasis<T::RealField, X, NumBasis=Dyn>,
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