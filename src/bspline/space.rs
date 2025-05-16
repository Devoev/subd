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
/// - [`N`] : Number of components of the basis functions.
/// For scalar valued functions (e.g. B-Splines) equal to `1`,
/// for vector valued functions equal to the dimension of the embedding space.
#[derive(Debug, Clone)]
pub struct SplineSpace<T: ComplexField, X, const N: usize, B: BsplineBasis<T::RealField, X, N>> {
    /// Set of basis functions spanning this function space.
    pub basis: B,
    
    phantom_data: PhantomData<(T, X)>
}

/// Space of univariate B-Splines.
pub type Splines1<T> = SplineSpace<T, T, 1, DeBoor<T>>;

/// Space of bivariate B-Splines.
pub type Splines2<T> = SplineSpace<T, SVector<T, 2>, 1, MultiDeBoor<T, 2>>;

/// Space of trivariate B-Splines.
pub type Splines3<T> = SplineSpace<T, SVector<T, 3>, 1, MultiDeBoor<T, 3>>;

impl <T: RealField, X, const N: usize, B: BsplineBasis<T, X, N>> SplineSpace<T, X, N, B> {
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
    pub fn linear_combination<const M: usize, Nc>(&self, coeffs: OMatrix<T, Const<M>, Nc>) -> Spline<T, X, B, M, N, Nc> 
        where Nc: Dim,
              DefaultAllocator: Allocator<Const<M>, Nc>
    {
        Spline::new(coeffs, self)
    }
}