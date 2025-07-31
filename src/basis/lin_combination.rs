use crate::basis::error::CoeffsSpaceDimError;
use crate::basis::eval::EvalBasis;
use crate::basis::local::{FindElem, LocalBasis};
use crate::basis::space::Space;
use crate::basis::traits::Basis;
use itertools::Itertools;
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
    where T: ComplexField,
          B: EvalBasis<T::RealField, X, NumBasis=Dyn>,
          DefaultAllocator: Allocator<B::NumComponents, B::NumBasis>,
          DefaultAllocator: Allocator<B::NumComponents>,
{
    /// Evaluates this linear combination at the parametric point `x`,
    /// by calculating `c[0] * b[0](x) + ... + c[n] * b[n](x)`.
    pub fn eval(&self, x: X) -> OVector<T, B::NumComponents> {
        let b= self.space.basis.eval(x);
        // todo: is there another way to calculate this without using .map ?
        //  Maybe add custom cast function?
        b.map(|bi| T::from_real(bi)) * &self.coeffs
    }
}

impl <'a, T, X, B, const D: usize> LinCombination<'a, T, X, B, D>
    where T: ComplexField,
          X: Copy,
          B: LocalBasis<T::RealField, X>,
          B::ElemBasis: Basis<NumBasis=Dyn>,
          DefaultAllocator: Allocator<B::NumComponents, <B::ElemBasis as Basis>::NumBasis>,
          DefaultAllocator: Allocator<B::NumComponents>,
{
    /// Evaluates the linear combination on the given `elem` at the parametric point `x`.
    pub fn eval_on_elem(&self, elem: &B::Elem, x: X) -> OVector<T, B::NumComponents> {
        let (b, idx) = self.space.eval_on_elem_with_idx(elem, x);
        let c = self.coeffs.select_rows(idx.collect_vec().iter()); // todo: remove collect
        b.map(|bi| T::from_real(bi)) * c
    }
}

impl <'a, T, X, B, const D: usize> LinCombination<'a, T, X, B, D>
where T: ComplexField,
      X: Copy,
      B: FindElem<T::RealField, X>,
      B::ElemBasis: Basis<NumBasis=Dyn>,
      DefaultAllocator: Allocator<B::NumComponents, <B::ElemBasis as Basis>::NumBasis>,
      DefaultAllocator: Allocator<B::NumComponents>,
{
    /// Evaluates the linear combination at the parametric point `x`.
    /// This is done by finding the local element in which `x` is, which can potentially be expensive.
    pub fn eval_local(&self, x: X) -> OVector<T, B::NumComponents>{
        let (b, idx) = self.space.eval_local_with_idx(x);
        let c = self.coeffs.select_rows(idx.collect_vec().iter()); // todo: remove collect
        b.map(|bi| T::from_real(bi)) * c
    }
}