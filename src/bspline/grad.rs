use std::marker::PhantomData;
use itertools::Itertools;
use nalgebra::{Const, Dyn, Matrix, OMatrix, RealField};
use crate::bspline::basis::BsplineBasis;
use crate::bspline::de_boor::DeBoorMulti;

/// Gradients of a [`BsplineBasis`].
pub struct BasisGrad<T: RealField, X, B: BsplineBasis<T::RealField, X, 1>> {
    /// Scalar basis.
    pub basis: B,
    
    _phantom_data: PhantomData<(T, X)>,
}

impl <T: RealField + Copy, const D: usize> BasisGrad<T, [T; D], DeBoorMulti<T, D>> {
    
    pub fn new(basis: DeBoorMulti<T, D>) -> Self {
        BasisGrad { basis, _phantom_data: PhantomData }
    }
    
    /// Evaluates the nonzero gradients of basis functions at the parametric point `x`.
    /// Each row in the result matrix corresponds to the values of the gradient of a single basis function.
    pub fn eval(&self, x: [T; D]) -> OMatrix<T, Dyn, Const<D>> {
        let cols = (0..D).map(|du| self.basis.eval_deriv_multi_prod(x, du).0).collect_vec();
        Matrix::from_columns(&cols)
    }
}