use crate::bspline::local_basis::LocalBsplineBasis;
use crate::knots::error::OutsideKnotRangeError;
use crate::knots::knot_span::KnotSpan;
use crate::knots::knot_vec::KnotVec;
use nalgebra::RealField;
use crate::cells::bezier_elem::BezierElem;

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

impl <T: RealField> BsplineBasis<T> {
    /// Constructs a new [`BsplineBasis`] from the given `knots`, `num_basis` and `degree`.
    pub fn new(knots: KnotVec<T>, num_basis: usize, degree: usize) -> Self {
        Self { knots, num_basis, degree }
    }
}

impl <T: RealField + Copy> BsplineBasis<T> {
    /// Finds the knot span for the parametric value `t` using [`KnotSpan::find`].
    pub(crate) fn find_span(&self, t: T) -> Result<KnotSpan, OutsideKnotRangeError> {
        KnotSpan::find(&self.knots, self.num_basis, t)
    }
    
    /// Finds the knot span for the 1D [`BezierElem`] `elem`.
    pub(crate) fn find_span_by_elem(&self, elem: BezierElem<T, 1, 1>) -> Result<KnotSpan, OutsideKnotRangeError> {
        self.find_span(elem.ref_elem.a.x)
    }
    
    /// Returns the [`LocalBsplineBasis`] for the given 1D `elem`.
    pub fn local_basis(&self, elem: BezierElem<T, 1, 1>) -> LocalBsplineBasis<T> {
        let span = self.find_span_by_elem(elem).unwrap();
        LocalBsplineBasis::new(&self.knots, self.degree, span)
    }
}