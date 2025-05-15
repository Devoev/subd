use std::marker::PhantomData;
use nalgebra::{stack, Dyn, OMatrix, RealField, U2};
use crate::bspline::basis::BsplineBasis;

/// Cartesian product of two scalar bases [`B1`] and [`B2`].
#[derive(Debug, Clone, Copy)]
pub struct Prod<T: RealField, X, B1: BsplineBasis<T, X>, B2: BsplineBasis<T, X>> {
    /// First basis.
    b1: B1,

    /// Second basis.
    b2: B2,

    _phantom_data: PhantomData<(T, X)>,
}

#[allow(clippy::toplevel_ref_arg)]
impl <T: RealField, X: Copy, B1: BsplineBasis<T, X>, B2: BsplineBasis<T, X>> Prod<T, X, B1, B2> {
    /// Constructs a new [`Prod`] from the bases `b1` and `b2`.
    pub fn new(b1: B1, b2: B2) -> Self {
        Prod { b1, b2, _phantom_data: Default::default() }
    }
    
    pub fn eval_nonzero(&self, x: X) -> OMatrix<T, Dyn, U2> {
        let (b1, _) = self.b1.eval_nonzero(x);
        let (b2, _) = self.b2.eval_nonzero(x);
        stack![b1, 0; 
               0, b2]
    }
}