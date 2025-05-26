use std::iter::zip;
use itertools::{izip, Itertools};
use crate::basis::global::GlobalBasis;
use crate::bspline::local_basis::{BsplineBasisLocal, MultiBsplineBasisLocal};
use crate::cells::hyper_rectangle::HyperRectangle;
use crate::knots::error::OutsideKnotRangeError;
use crate::knots::knot_span::KnotSpan;
use crate::knots::knot_vec::KnotVec;
use nalgebra::{vector, RealField};
use crate::basis::tensor_prod::MultiProd;

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
    
    /// Finds the knot span for the 1D [`HyperRectangle`] `elem`.
    pub(crate) fn find_span_by_elem(&self, elem: &HyperRectangle<T, 1>) -> Result<KnotSpan, OutsideKnotRangeError> {
        self.find_span(elem.a.x)
    }
}
impl <T: RealField + Copy> GlobalBasis<T, T, 1> for BsplineBasis<T> {
    type Elem = HyperRectangle<T, 1>;
    type LocalBasis<'a> = BsplineBasisLocal<'a, T>;

    fn local_basis(&self, elem: &Self::Elem) -> Self::LocalBasis<'_> {
        let span = self.find_span_by_elem(elem).unwrap();
        BsplineBasisLocal::new(&self.knots, self.degree, span)
    }
}

// todo: replace this impl with a generic impl for every GlobalBasis

/// Basis of [`D`]-variate B-Splines on an entire knot vector.
pub type MultiBsplineBasis<T, const D: usize> = MultiProd<T, BsplineBasis<T>, D>;

impl<T: RealField + Copy, const D: usize> GlobalBasis<T, [T; D], 1> for MultiBsplineBasis<T, D> {
    type Elem = HyperRectangle<T, D>;
    type LocalBasis<'a> = MultiBsplineBasisLocal<'a, T, D>;

    fn local_basis(&self, elem: &Self::Elem) -> Self::LocalBasis<'_> {
        let bases = izip!(&self.bases, &elem.a, &elem.b)
            .map(|(b, &ai, &bi)| {
                let interval = HyperRectangle::new(vector![ai], vector![bi]);
                b.local_basis(&interval)
            }).collect_array().unwrap();
        MultiBsplineBasisLocal::new(bases)
    }
}