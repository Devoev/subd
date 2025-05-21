use crate::bspline::basis::{BsplineBasis, ScalarBasis};
use crate::bspline::tensor_prod::{MultiProd, Prod};
use crate::knots::knot_span::KnotSpan;
use crate::knots::knot_vec::KnotVec;
use itertools::{chain, Itertools};
use nalgebra::{Const, DVector, DimNameAdd, DimNameSum, Dyn, OMatrix, RealField, U1};
use std::iter::zip;
use std::ops::RangeInclusive;
use std::vec;

/// De-Boor algorithm for the computation of the B-Spline basis
/// of dimension `n` and degree `p`.
#[derive(Debug, Clone)]
pub struct DeBoor<T: RealField> {
    /// The vector of knot values.
    pub(crate) knots: KnotVec<T>,
    
    /// Number of basis functions.
    pub n: usize,
    
    /// Degree of basis functions.
    pub p: usize
}


/// Tensor product generalization of [de Boors algorithm](DeBoor) for the
/// computation of a [`D`]-variate B-Spline basis.
pub type DeBoorMulti<T, const D: usize> = MultiProd<T, DeBoor<T>, D>;

/// Bivariate tensor product [de Boor algorithm](DeBoor).
pub type DeBoorBi<T> = Prod<T, DeBoor<T>>;

impl<T : RealField + Copy> DeBoor<T> {

    /// Constructs a new [`DeBoor`].
    /// 
    /// If the values of `n` and `p` don't match the length of the knots,
    /// [`None`] is returned.
    pub fn new(knots: KnotVec<T>, n: usize, p: usize) -> Option<Self> {
        (knots.len() == n + p + 1).then_some(DeBoor { knots, n, p })
    }
    
    /// Constructs an open [`DeBoor`] with given `internal` knot values.
    /// 
    /// If internal knots are not inside `(0,1)`, [`None`] is returned.
    pub fn open(internal: KnotVec<T>, n: usize, p: usize) -> Option<Self> {
        let knots = chain!(
            std::iter::repeat_n(T::zero(), p+1),
            internal.0,
            std::iter::repeat_n(T::one(), p+1)
        ).collect();
        
        Self::new(KnotVec(knots), n, p)
    }

    /// Constructs an open uniform [`DeBoor`].
    pub fn open_uniform(n: usize, p: usize) -> Self {
        let internal = KnotVec::uniform(n-p+1);
        let knots = chain!(
            std::iter::repeat_n(T::zero(), p),
            internal.0,
            std::iter::repeat_n(T::one(), p)
        ).collect();
        
        DeBoor { knots: KnotVec(knots), n, p }
    }

    /// Finds the knot span for the parametric value `t` using [`KnotSpan::find`].
    pub(crate) fn find_span(&self, t: T) -> Result<KnotSpan, ()> {
        KnotSpan::find(&self.knots, self.n, t)
    }

    /// Evaluates the basis functions inside the given `span` at the parametric point `t`.
    pub(crate) fn eval_with_span(&self, t: T, span: KnotSpan) -> DVector<T> {
        let knots = &self.knots;
        let mut left = vec![T::zero(); self.p + 1];
        let mut right = vec![T::zero(); self.p + 1];
        let mut b = DVector::zeros(self.p + 1);
        b[0] = T::one();

        for i in 1..=self.p {
            left[i] = t - knots[span.0 - i + 1];
            right[i] = knots[span.0 + i] - t;
            let mut saved = T::zero();

            for j in 0..i {
                let tmp = b[j] / (right[j+1] + left[i-j]);
                b[j] = saved + right[j+1]*tmp;
                saved = left[i-j]*tmp;
            }
            b[i] = saved;
        }
        b
    }

    /// Evaluates the [`K`] derivatives of the basis functions
    /// inside the given `span` at the parametric point `t`.
    pub(crate) fn eval_derivs_with_span<const K: usize>(&self, t: T, span: KnotSpan) -> OMatrix<T, DimNameSum<Const<K>, U1>, Dyn>
        where Const<K>: DimNameAdd<U1>
    {
        let knots = &self.knots;
        let mut ndu = vec![vec![T::zero(); self.p + 1]; self.p + 1];
        let mut left = vec![T::zero(); self.p + 1];
        let mut right = vec![T::zero(); self.p + 1];

        ndu[0][0] = T::one();

        for j in 1..=self.p {
            left[j] = t - knots[span.0 + 1 - j];
            right[j] = knots[span.0 + j] - t;

            let mut saved = T::zero();
            for r in 0..j {
                // lower triangle
                ndu[j][r] = right[r + 1] + left[j - r];
                let temp = ndu[r][j - 1] / ndu[j][r];

                // upper triangle
                ndu[r][j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            ndu[j][j] = saved;
        }

        let mut ders = OMatrix::<T, DimNameSum<Const<K>, U1>, Dyn>::zeros(self.p + 1);
        let mut a = vec![vec![T::zero(); self.p + 1]; 2];

        // load the basis functions
        for j in 0..=self.p {
            ders[(0,j)] = ndu[j][self.p];
        }

        let idegree = self.p as isize;
        let n = K as isize;

        // compute the derivatives
        for r in 0..=idegree {
            // alternate rows in array a
            let mut s1 = 0;
            let mut s2 = 1;
            a[0][0] = T::one();

            // loop to compute the kth derivative
            for k in 1..=n {
                let mut d = T::zero();
                let rk = r - k;
                let pk = idegree - k;

                if r >= k {
                    a[s2][0] = a[s1][0] / ndu[(pk + 1) as usize][rk as usize];
                    d = a[s2][0] * ndu[rk as usize][pk as usize];
                }

                let j1 = if rk >= -1 { 1 } else { -rk };
                let j2 = if r - 1 <= pk { k - 1 } else { idegree - r };

                for j in j1..=j2 {
                    a[s2][j as usize] = (a[s1][j as usize] - a[s1][j as usize - 1])
                        / ndu[(pk + 1) as usize][(rk + j) as usize];
                    d += a[s2][j as usize] * ndu[(rk + j) as usize][pk as usize];
                }

                let uk = k as usize;
                let ur = r as usize;
                if r <= pk {
                    a[s2][uk] = -a[s1][(k - 1) as usize] / ndu[(pk + 1) as usize][ur];
                    d += a[s2][uk] * ndu[ur][pk as usize];
                }

                ders[(uk,ur)] = d;

                // switch rows
                std::mem::swap(&mut s1, &mut s2);
            }
        }

        let mut acc = idegree;
        for k in 1..=n {
            for j in 0..=idegree {
                ders[(k as usize, j as usize)] *= T::from_isize(acc).unwrap();
            }
            acc *= idegree - k;
        }
        ders
    }
}

impl <T: RealField + Copy> BsplineBasis<T, T, 1> for DeBoor<T> {
    type NonzeroIndices = RangeInclusive<usize>;

    fn num_basis(&self) -> usize {
        self.n
    }

    fn eval_nonzero(&self, x: T) -> (DVector<T>, Self::NonzeroIndices) {
        let span = self.find_span(x).unwrap();
        (self.eval_with_span(x, span), span.nonzero_indices(self.p))
    }
}

impl <T: RealField + Copy> ScalarBasis<T, T> for DeBoor<T> {
    fn eval_derivs_nonzero<const K: usize>(&self, x: T) -> (OMatrix<T, DimNameSum<Const<K>, U1>, Dyn>, Self::NonzeroIndices)
    where
        Const<K>: DimNameAdd<U1>
    {
        let span = self.find_span(x).unwrap();
        (self.eval_derivs_with_span(x, span), span.nonzero_indices(self.p))
    }
}

impl<T: RealField + Copy, const D: usize> DeBoorMulti<T, D> {
    /// Constructs an open [`DeBoorMulti`] with `n[i] + p[i] + 1` for each direction.
    pub fn open_uniform(n: [usize; D], p: [usize; D]) -> Self {
        let arr: [DeBoor<T>; D] = zip(n, p)
            .map(|(n, p)| DeBoor::open_uniform(n, p))
            .collect_vec()
            .try_into()
            .unwrap();

        DeBoorMulti::new(arr)
    }
}

impl<T: RealField + Copy, const D : usize> DeBoorMulti<T, D> {
    /// Return the degrees of basis functions per parametric direction.
    pub fn degrees(&self) -> [usize; D] {
        self.bases.iter().map(|b| b.p).collect_array().unwrap()
    }
}
