use std::ops::RangeInclusive;
use itertools::{izip, Itertools};
use crate::basis::local::LocalBasis;
use crate::basis::tensor_prod::MultiProd;
use crate::bspline::local_basis::BsplineBasisLocal;
use crate::cells::hyper_rectangle::HyperRectangle;
use crate::knots::error::OutsideKnotRangeError;
use crate::knots::knot_span::KnotSpan;
use crate::knots::knot_vec::KnotVec;
use nalgebra::{Const, Dyn, OMatrix, RealField, U1};
use crate::basis::eval::{EvalBasis, EvalGrad};
use crate::basis::traits::Basis;

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

impl <T: RealField, const D: usize> MultiBsplineBasis<T, D> {
    /// Constructs a new [`MultiBsplineBasis`] from the given `knots`, `num_bases` and `degrees`
    /// for each parametric direction.
    pub fn from_knots(knots: [KnotVec<T>; D], num_bases: [usize; D], degrees: [usize; D]) -> Self {
        let bases = izip!(knots, num_bases, degrees)
            .map(|(k, n, p)| BsplineBasis::new(k, n, p))
            .collect_array().unwrap();
        MultiBsplineBasis::new(bases)
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

impl<T: RealField> Basis for BsplineBasis<T> {
    type NumBasis = Dyn;
    type NumComponents = U1;

    fn num_basis(&self) -> usize {
        self.num_basis
    }

    fn num_basis_generic(&self) -> Self::NumBasis {
        Dyn(self.num_basis)
    }

    fn num_components(&self) -> usize {
        1
    }

    fn num_components_generic(&self) -> Self::NumComponents {
        U1
    }
}

// impl<T: RealField + Copy> EvalBasis<T, T> for BsplineBasis<T> {
//     fn eval(&self, x: T) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
//         // todo: possibly change this, to return the full sized vector and not the local one
//         let span = self.find_span(x).unwrap();
//         self.elem_basis(&span).eval(x)
//     }
// }
//
// impl <T: RealField + Copy> EvalGrad<T, T, 1> for BsplineBasis<T> {
//     fn eval_grad(&self, x: T) -> OMatrix<T, Const<1>, Dyn> {
//         // todo: possibly change this, to return the full sized vector and not the local one
//         let span = self.find_span(x).unwrap();
//         self.elem_basis(&span).eval_grad(x)
//     }
// }

impl <T: RealField + Copy> LocalBasis<T, T> for BsplineBasis<T> {
    type Elem = KnotSpan;
    type ElemBasis = BsplineBasisLocal<T>;
    type GlobalIndices = RangeInclusive<usize>;

    fn find_elem(&self, x: T) -> Self::Elem {
        self.find_span(x).unwrap()
    }

    fn elem_basis(&self, elem: &Self::Elem) -> Self::ElemBasis {
        BsplineBasisLocal::new(self.knots.clone(), self.degree, *elem)
    }

    fn global_indices(&self, local_basis: &Self::ElemBasis) -> Self::GlobalIndices {
        local_basis.span.nonzero_indices(self.degree)
    }
}