use std::marker::PhantomData;
use nalgebra::RealField;
use crate::bspline::basis::BsplineBasis;

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
}