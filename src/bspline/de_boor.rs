use crate::basis::cart_prod;
use crate::basis::local::LocalBasis;
use crate::basis::tensor_prod::MultiProd;
use crate::basis::traits::Basis;
use crate::bspline::de_boor_span::DeBoorSpan;
use crate::cells::hyper_rectangle::HyperRectangle;
use crate::knots::error::OutsideKnotRangeError;
use crate::knots::knot_span::KnotSpan;
use crate::knots::knot_vec::KnotVec;
use itertools::{izip, Itertools};
use nalgebra::{Dyn, RealField, U1};
use std::ops::RangeInclusive;

/// Scalar univariate B-Spline basis functions on a [`KnotVec<T>`].
/// 
/// The basis evaluation is done using the [De-Boor algorithm](https://en.wikipedia.org/wiki/De_Boor%27s_algorithm).
#[derive(Clone, Debug)]
pub struct DeBoor<T> {
    /// The vector of knot values.
    pub(crate) knots: KnotVec<T>,

    /// Number of basis functions.
    pub num_basis: usize,

    /// Degree of basis functions.
    pub degree: usize
}

/// Basis of [`D`]-variate B-Splines on an entire knot vector.
pub type MultiDeBoor<T, const D: usize> = MultiProd<T, DeBoor<T>, D>;

/// Basis of 2D vector valued B-Splines.
pub type DeBoorVec2d<T> = cart_prod::Prod<T, MultiDeBoor<T, 2>, MultiDeBoor<T, 2>>;

impl <T: RealField> DeBoor<T> {
    /// Constructs a new [`DeBoor`] from the given `knots`, `num_basis` and `degree`.
    pub fn new(knots: KnotVec<T>, num_basis: usize, degree: usize) -> Self {
        Self { knots, num_basis, degree }
    }
}

impl <T: RealField, const D: usize> MultiDeBoor<T, D> {
    /// Constructs a new [`MultiDeBoor`] from the given `knots`, `num_bases` and `degrees`
    /// for each parametric direction.
    pub fn from_knots(knots: [KnotVec<T>; D], num_bases: [usize; D], degrees: [usize; D]) -> Self {
        let bases = izip!(knots, num_bases, degrees)
            .map(|(k, n, p)| DeBoor::new(k, n, p))
            .collect_array().unwrap();
        MultiDeBoor::new(bases)
    }
}

impl <T: RealField + Copy> DeBoor<T> {
    /// Finds the knot span for the parametric value `t` using [`KnotSpan::find`].
    pub(crate) fn find_span(&self, t: T) -> Result<KnotSpan, OutsideKnotRangeError> {
        KnotSpan::find(&self.knots, self.num_basis, t)
    }
    
    /// Finds the knot span for the 1D [`HyperRectangle`] `elem`.
    pub(crate) fn find_span_by_elem(&self, elem: &HyperRectangle<T, 1>) -> Result<KnotSpan, OutsideKnotRangeError> {
        self.find_span(elem.a.x)
    }
}

impl<T: RealField> Basis for DeBoor<T> {
    type NumBasis = Dyn;
    type NumComponents = U1;

    fn num_basis_generic(&self) -> Self::NumBasis {
        Dyn(self.num_basis)
    }
}

impl <T: RealField + Copy> LocalBasis<T, T> for DeBoor<T> {
    type Elem = KnotSpan;
    type ElemBasis = DeBoorSpan<T>;
    type GlobalIndices = RangeInclusive<usize>;

    fn find_elem(&self, x: T) -> Self::Elem {
        self.find_span(x).unwrap()
    }

    fn elem_basis(&self, elem: &Self::Elem) -> Self::ElemBasis {
        // todo: replace knots.clone() for efficiency
        DeBoorSpan::new(self.knots.clone(), self.degree, *elem)
    }

    fn global_indices(&self, local_basis: &Self::ElemBasis) -> Self::GlobalIndices {
        local_basis.span.nonzero_indices(self.degree)
    }
}