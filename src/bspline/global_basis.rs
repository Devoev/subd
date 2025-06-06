use std::ops::RangeInclusive;
use crate::basis::local::LocalBasis;
use crate::basis::tensor_prod::MultiProd;
use crate::bspline::local_basis::BsplineBasisLocal;
use crate::cells::hyper_rectangle::HyperRectangle;
use crate::knots::error::OutsideKnotRangeError;
use crate::knots::knot_span::KnotSpan;
use crate::knots::knot_vec::KnotVec;
use nalgebra::RealField;
use crate::basis::traits::NumBasis;

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

/// Basis of [`D`]-variate B-Splines on an entire knot vector.
pub type MultiBsplineBasis<T, const D: usize> = MultiProd<T, BsplineBasis<T>, D>;

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

impl<T: RealField + Copy> NumBasis for BsplineBasis<T> {
    fn num_basis(&self) -> usize {
        self.num_basis
    }
}

impl <T: RealField + Copy> LocalBasis<T, T> for BsplineBasis<T> {
    type Elem = HyperRectangle<T, 1>;
    type ElemBasis = BsplineBasisLocal<T>;
    type GlobalIndices = RangeInclusive<usize>;
    
    fn elem_basis(&self, elem: &Self::Elem) -> Self::ElemBasis {
        let span = self.find_span_by_elem(elem).unwrap();
        BsplineBasisLocal::new(self.knots.clone(), self.degree, span)
    }

    fn global_indices(&self, local_basis: &Self::ElemBasis) -> Self::GlobalIndices {
        local_basis.span.nonzero_indices(self.degree)
    }
}