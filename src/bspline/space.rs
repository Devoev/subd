use crate::bspline::basis::BsplineBasis;
use crate::bspline::de_boor::{DeBoor, MultiDeBoor};
use nalgebra::{ComplexField, Const, DefaultAllocator, Dim, Dyn, OMatrix, RealField, SVector};
use std::marker::PhantomData;
use nalgebra::allocator::Allocator;
use crate::bspline::spline::Spline;

/// Function space spanned by the B-Spline basis functions.
/// 
/// # Type parameters
/// - [`T`]: Scalar type for coefficients.
/// - [`X`]: Type of parametric values in the reference domain.
/// - [`B`]: B-Spline basis.
#[derive(Debug, Clone)]
pub struct SplineSpace<T: ComplexField, X, B: BsplineBasis<T::RealField, X>> {
    /// Set of basis functions spanning this function space.
    pub basis: B,
    
    phantom_data: PhantomData<(T, X)>
}

/// Space of univariate B-Splines.
pub type Splines1<T> = SplineSpace<T, T, DeBoor<T>>;

/// Space of bivariate B-Splines.
pub type Splines2<T> = SplineSpace<T, SVector<T, 2>, MultiDeBoor<T, 2>>;

/// Space of trivariate B-Splines.
pub type Splines3<T> = SplineSpace<T, SVector<T, 3>, MultiDeBoor<T, 3>>;

impl <T: RealField, X, B: BsplineBasis<T, X>> SplineSpace<T, X, B> {
    /// Constructs a new [`SplineSpace`] from the given `basis`.
    pub fn new(basis: B) -> Self {
        Self { basis, phantom_data: PhantomData }
    }
    
    /// Returns the dimension of this space, i.e. the number of basis functions in its basis.
    pub fn dim(&self) -> usize {
        self.basis.len()
    }

    /// Calculates the linear combination of the given `coeffs` with the basis function of this space,
    /// and returns the resulting [`Spline`] function.
    pub fn linear_combination<const M: usize, N>(&self, coeffs: OMatrix<T, Const<M>, N>) -> Spline<T, X, B, M, N> 
        where N: Dim, 
              DefaultAllocator: Allocator<Const<M>, N>
    {
        Spline::new(coeffs, self)
    }
}