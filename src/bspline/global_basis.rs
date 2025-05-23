use crate::bspline::local_basis::LocalBsplineBasis;
use crate::knots::error::OutsideKnotRangeError;
use crate::knots::knot_span::KnotSpan;
use crate::knots::knot_vec::KnotVec;
use nalgebra::RealField;

/// B-Spline basis on an entire knot vector.
#[derive(Clone, Debug)]
pub struct BsplineBasis<T> {
    /// The vector of knot values.
    pub(crate) knots: KnotVec<T>,

    /// Number of basis functions.
    pub num_basis: usize,

    /// Degree of basis functions.
    pub degree: usize
}

impl <T: RealField + Copy> BsplineBasis<T> {
    /// Finds the knot span for the parametric value `t` using [`KnotSpan::find`].
    pub(crate) fn find_span(&self, t: T) -> Result<KnotSpan, OutsideKnotRangeError> {
        KnotSpan::find(&self.knots, self.num_basis, t)
    }
    
    /// Returns the [`LocalBsplineBasis`] for the given `span`.
    pub fn local_basis(&self, span: KnotSpan) -> LocalBsplineBasis<T> {
        LocalBsplineBasis::new(&self.knots, self.degree, span)
    }
}