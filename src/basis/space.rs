use crate::basis::traits::{Basis};
use std::marker::PhantomData;
use nalgebra::{DVector, RealField};
use crate::basis::lin_combination::LinCombination;

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