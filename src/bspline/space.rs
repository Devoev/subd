use crate::bspline::basis::BsplineBasis;
use crate::bspline::control_points::ControlPoints;
use itertools::Itertools;
use nalgebra::{Const, Dim, RealField, SVector, Storage};
use std::marker::PhantomData;

/// Function space spanned by the B-Spline basis functions.
#[derive(Debug, Clone)]
pub struct BsplineSpace<T: RealField, X, B: BsplineBasis<T, X>> {
    /// Set of basis functions spanning this function space.
    pub basis: B,
    
    phantom_data: PhantomData<(T, X)>
}

impl <T: RealField, X, B: BsplineBasis<T, X>> BsplineSpace<T, X, B> {
    /// Constructs a new [`BsplineSpace`] from the given `basis`.
    pub fn new(basis: B) -> Self {
        Self { basis, phantom_data: PhantomData }
    }
    
    /// Returns the dimension of this space, i.e. the number of basis functions in its basis.
    pub fn dim(&self) -> usize {
        self.basis.len()
    }

    /// Evaluates the B-Spline function `f = ∑ cⁱ bᵢ` 
    /// represented by the coefficients `cⁱ ∈ ℝᵈ`
    /// at the parametric point `x`.
    pub fn eval_coeffs<const D: usize, Nc, S>(&self, c: &ControlPoints<T, D, Nc, S>, x: X) -> SVector<T, D>
        where Nc: Dim,
            S: Storage<T, Const<D>, Nc>
    {
        let (b, idx) = self.basis.eval_nonzero(x);
        let c = c.get_nonzero(idx.collect_vec().iter());
        c.coords * b
    }
}