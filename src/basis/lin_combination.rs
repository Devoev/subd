use itertools::Itertools;
use crate::basis::space::Space;
use crate::basis::traits::{Basis, NumBasis};
use nalgebra::{ComplexField, DVector, RealField, SVector};
use crate::basis::local::LocalBasis;

/// Linear combination of coefficients with basis functions.
pub struct LinCombination<'a, T: ComplexField, X, const N: usize, B> {
    /// Coefficients vector.
    pub coeffs: DVector<T>,

    /// Space of basis functions.
    pub space: &'a Space<T::RealField, X, N, B>,
}

impl <'a, T: ComplexField, X, const N: usize, B: NumBasis> LinCombination<'a, T, X, N, B> {
    /// Constructs a new [`LinCombination`] from the given `coeffs` and `space`.
    pub fn new(coeffs: DVector<T>, space: &'a Space<T::RealField, X, N, B>) -> Self {
        assert_eq!(coeffs.ncols(), space.dim(),
                   "The number of coefficients (is {}) must match the dimension of the space (is {})",
                   coeffs.ncols(), space.dim());

        LinCombination { coeffs, space }
    }
}

impl <'a, T, X, const N: usize, B> LinCombination<'a, T, X, N, B>
    where T: RealField, // todo: replace this with ComplexField and fix c * b in eval
          B: Basis<T::RealField, X, N>,
{
    /// Evaluates this linear combination at the parametric point `x`,
    /// by calculating `c[0] * b[0](x) + ... + c[n] * b[n](x)`.
    pub fn eval(&self, x: X) -> SVector<T, N> {
        let b= self.space.basis.eval(x);
        b * &self.coeffs
    }
}

impl <'a, T, X, const N: usize, B> LinCombination<'a, T, X, N, B>
    where T: RealField, // todo: replace this with ComplexField and fix c * b in eval
          B: LocalBasis<T::RealField, X, N>,
{
    // todo: this is ugly. change signature and how to get/provide the element
    //  possibly add method to LocalBasis
    /// Evaluates this linear combination at the parametric point `x`,
    /// by calculating `c[0] * b[0](x) + ... + c[n] * b[n](x)`.
    pub fn eval_on_elem(&self, elem: &B::Elem, x: X) -> SVector<T, N> {
        let elem_basis = self.space.basis.elem_basis(elem);
        let idx = self.space.basis.global_indices(&elem_basis);
        let b = elem_basis.eval(x);
        let c = self.coeffs.select_rows(idx.collect_vec().iter());
        b * c
    }
}