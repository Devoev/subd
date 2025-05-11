use crate::bspline::basis::BsplineBasis;
use crate::bspline::space::SplineSpace;
use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::{ComplexField, Const, DefaultAllocator, Dim, OMatrix, RealField, SVector};

// todo: having N as a generic parameter instead of Dyn is important,
//  in order to allow for matrices created by the matrix![] macro to work with this struct.
//  Otherwise you'd always need to convert it to a dynamic matrix

/// A spline function represented by coefficients `cⁱ ∈ ℝᵐ`
/// as `f = ∑ cⁱ bᵢ`, where the `bᵢ` are the B-Splines of the basis [`B`].
///
/// # Type parameters
/// - [`T`]: Scalar type for coefficients.
/// - [`X`]: Type of parametric values in the reference domain.
/// - [`B`]: B-Spline basis.
/// - [`M`]: Fixed size dimension of embedding Euclidean space, i.e. number of rows of coeffs matrix.
/// - [`N`]: Number (fixed size or dynamic) of coefficients, i.e. number of columns of coeffs matrix.
#[derive(Debug, Clone)]
pub struct Spline<'a, T, X, B, const M: usize, N>
    where T: ComplexField,
          B: BsplineBasis<T::RealField, X>,
          N: Dim,
          DefaultAllocator: Allocator<Const<M>, N>
{
    /// Coefficients representing this spline.
    pub coeffs: OMatrix<T, Const<M>, N>,

    /// The spline space this spline belongs to.
    space: &'a SplineSpace<T::RealField, X, B>,
}

impl <'a, T, X, B, const M: usize, N> Spline<'a, T, X, B, M, N>
    where T: RealField, // todo: replace this with ComplexField and fix c * b in eval
          B: BsplineBasis<T::RealField, X>,
          N: Dim,
          DefaultAllocator: Allocator<Const<M>, N>
{
    /// Constructs a new [`SplineSpace`] from the given `coeffs` and `space`.
    pub fn new(coeffs: OMatrix<T, Const<M>, N>, space: &'a SplineSpace<T::RealField, X, B>) -> Self {
        assert_eq!(coeffs.ncols(), space.dim(),
                   "The number of coefficients (is {}) must match the dimension of the spline space (is {})",
                   coeffs.ncols(), space.dim());

        Spline { coeffs, space }
    }

    /// Evaluates this spline at the parametric point `x`,
    /// by calculating the linear combination `c[0] * b[0](x) + ... + c[n] * b[n](x)`.
    pub fn eval(&self, x: X) -> SVector<T, M> {
        let (b, idx) = self.space.basis.eval_nonzero(x);
        let c = self.coeffs.select_columns(idx.collect_vec().iter());
        c * b
    }
}