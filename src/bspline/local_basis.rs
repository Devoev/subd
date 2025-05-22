use crate::basis::local::{LocalBasis, LocalHgradBasis};
use crate::bspline::de_boor::DeBoor;
use crate::knots::knot_span::KnotSpan;
use nalgebra::{Const, Dyn, OMatrix, RealField};
use std::ops::RangeInclusive;

/// Local B-Spline basis inside a knot span.
#[derive(Debug, Clone)]
pub struct LocalBsplineBasis<'a, T: RealField> {
    /// Global B-Spline basis.
    pub global_basis: &'a DeBoor<T>, // todo: possibly replace with local knot vector view and degree?

    /// Knot span of the local basis.
    span: KnotSpan,
}

impl<'a, T: RealField + Copy> LocalBasis<T, T, 1> for LocalBsplineBasis<'a, T> {
    type LinIndices = RangeInclusive<usize>;

    fn num_basis(&self) -> usize {
        self.global_basis.p + 1
    }

    fn global_indices(&self) -> Self::LinIndices {
        self.span.nonzero_indices(self.global_basis.p)
    }

    fn eval(&self, x: T) -> OMatrix<T, Const<1>, Dyn> {
        self.global_basis.eval_with_span(x, self.span).transpose()
    }
}

impl<'a, T: RealField + Copy> LocalHgradBasis<T, T, 1> for LocalBsplineBasis<'a, T> {
    fn eval_grad(&self, x: T) -> OMatrix<T, Const<1>, Dyn> {
        let derivs = self.global_basis.eval_derivs_with_span::<1>(x, self.span);
        derivs.row(1).into_owned()
    }
}