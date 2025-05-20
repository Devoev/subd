use crate::bspline::basis::BsplineBasis;
use crate::bspline::space::SplineSpace;
use crate::bspline::spline_geo::SplineGeo;
use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::{ComplexField, Const, DefaultAllocator, Dim, Dyn, OMatrix, RealField, SMatrix};

// todo: having Nc as a generic parameter instead of Dyn is important,
//  in order to allow for matrices created by the matrix![] macro to work with this struct.
//  Otherwise you'd always need to convert it to a dynamic matrix

// todo: should the M argument be removed?

/// A spline function represented by coefficients `c[i] ∈ ℝᵐ`
/// as `f = ∑ cⁱ[i]b[i]`, where the `bᵢ` are the B-Splines of the basis [`B`].
/// 
/// # Special cases
/// - [`SplineGeo`] for scalar valued basis functions (`N = 1`). 
///     Represents curves, surfaces, and solids in Euclidean space.
/// - [`SplineVector`] for scalar valued coefficients (`M = 1`).
///     Represents vector fields.
/// - [`SplineScalar`] for scalar valued coefficients and basis functions (`M = N = 1`).
///     Represents scalar fields.
///
/// # Type parameters
/// - [`T`]: Scalar type for coefficients.
/// - [`X`]: Type of parametric values in the reference domain.
/// - [`B`]: B-Spline basis.
/// - [`M`]: Fixed size dimension of each coefficient, i.e. number of rows of coeffs matrix.
/// - [`N`]: Fixed size of components of basis functions, i.e. number of columns of [`BsplineBasis::eval_nonzero`].
/// - [`Nc`]: Number (fixed size or dynamic) of coefficients, i.e. number of columns of coeffs matrix. [`Dyn`] by default.
#[derive(Debug, Clone)]
pub struct Spline<'a, T, X, B, const M: usize, const N: usize, Nc = Dyn>
    where T: ComplexField,
          B: BsplineBasis<T::RealField, X, N>,
          Nc: Dim,
          DefaultAllocator: Allocator<Const<M>, Nc>
{
    /// Coefficients representing this spline.
    pub coeffs: OMatrix<T, Const<M>, Nc>,

    /// The spline space this spline belongs to.
    pub space: &'a SplineSpace<T::RealField, X, N, B>,
}

/// A vector valued [`Spline`] function of [`N`] components.
pub type SplineVector<'a, T, X, B, const N: usize, Nc = Dyn> = Spline<'a, T, X, B, N, 1, Nc>;

/// A scalar valued [`Spline`] function.
pub type SplineScalar<'a, T, X, B, Nc = Dyn> = SplineVector<'a, T, X, B, 1, Nc>;

impl <'a, T, X, B, const M: usize, const N: usize, Nc> Spline<'a, T, X, B, M, N, Nc>
    where T: RealField, // todo: replace this with ComplexField and fix c * b in eval
          B: BsplineBasis<T::RealField, X, N>,
          Nc: Dim,
          DefaultAllocator: Allocator<Const<M>, Nc>
{
    /// Constructs a new [`SplineSpace`] from the given `coeffs` and `space`.
    pub fn new(coeffs: OMatrix<T, Const<M>, Nc>, space: &'a SplineSpace<T::RealField, X, N, B>) -> Self {
        assert_eq!(coeffs.ncols(), space.dim(),
                   "The number of coefficients (is {}) must match the dimension of the spline space (is {})",
                   coeffs.ncols(), space.dim());

        Spline { coeffs, space }
    }

    /// Evaluates this spline at the parametric point `x`,
    /// by calculating the linear combination `c[0] * b[0](x) + ... + c[n] * b[n](x)`.
    pub fn eval(&self, x: X) -> SMatrix<T, M, N> {
        let (b, idx) = self.space.basis.eval_nonzero(x);
        let c = self.coeffs.select_columns(idx.collect_vec().iter());
        c * b
    }
}