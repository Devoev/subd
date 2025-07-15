use crate::basis::eval::{EvalBasis, EvalDerivs, EvalGrad};
use crate::basis::tensor_prod::MultiProd;
use crate::basis::traits::Basis;
use crate::knots::knot_span::KnotSpan;
use crate::knots::knot_vec::KnotVec;
use nalgebra::{Const, DimNameAdd, DimNameSum, Dyn, OMatrix, RealField, RowDVector, U1};

/// Scalar univariate B-Spline basis functions, restricted to a local [`KnotSpan`].
#[derive(Debug, Clone)]
pub struct DeBoorSpan<T: RealField> {
    /// Global knot vector.
    knots: KnotVec<T>, // todo: replace with local knot vector copy

    /// Degree of basis functions.
    degree: usize,

    /// Knot span of the local basis.
    pub span: KnotSpan,
}

/// Basis of [`D`]-variate [local B-Splines](DeBoorSpan) on a local knot span.
pub type MultiDeBoorSpan<T, const D: usize> = MultiProd<T, DeBoorSpan<T>, D>;

impl <T: RealField> DeBoorSpan<T> {
    /// Constructs a new [`DeBoorSpan`] from the given `global_basis` and `span`.
    pub fn new(knots: KnotVec<T>, degree: usize, span: KnotSpan) -> Self {
        Self { knots, degree, span }
    }
}

impl<T: RealField> Basis for DeBoorSpan<T> {
    type NumBasis = Dyn;
    type NumComponents = U1;

    fn num_basis_generic(&self) -> Self::NumBasis {
        Dyn(self.degree + 1)
    }
}

impl<T: RealField + Copy> EvalBasis<T, T> for DeBoorSpan<T> {
    fn eval(&self, x: T) -> OMatrix<T, Const<1>, Dyn> {
        let knots = &self.knots;
        let span_idx = self.span.0;
        let p = self.degree;
        let mut left = vec![T::zero(); p + 1];
        let mut right = vec![T::zero(); p + 1];
        let mut b = RowDVector::zeros(p + 1);
        b[0] = T::one();

        for i in 1..=p {
            left[i] = x - knots[span_idx - i + 1];
            right[i] = knots[span_idx + i] - x;
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
}

impl <T: RealField + Copy> EvalDerivs<T, T> for DeBoorSpan<T> {
    fn eval_derivs<const K: usize>(&self, x: T) -> OMatrix<T, DimNameSum<Const<K>, U1>, Dyn>
    where
        Const<K>: DimNameAdd<U1>
    {
        let knots = &self.knots;
        let span = self.span;
        let p = self.degree;

        let mut ndu = vec![vec![T::zero(); p + 1]; p + 1];
        let mut left = vec![T::zero(); p + 1];
        let mut right = vec![T::zero(); p + 1];

        ndu[0][0] = T::one();

        for j in 1..=p {
            left[j] = x - knots[span.0 + 1 - j];
            right[j] = knots[span.0 + j] - x;

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

        let mut ders = OMatrix::<T, DimNameSum<Const<K>, U1>, Dyn>::zeros(p + 1);
        let mut a = vec![vec![T::zero(); p + 1]; 2];

        // load the basis functions
        for j in 0..=p {
            ders[(0,j)] = ndu[j][p];
        }

        let idegree = p as isize;
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

impl<T: RealField + Copy> EvalGrad<T, T, 1> for DeBoorSpan<T> {
    fn eval_grad(&self, x: T) -> OMatrix<T, Const<1>, Dyn> {
        let derivs = self.eval_derivs::<1>(x);
        derivs.row(1).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use itertools::izip;
    use nalgebra::dvector;

    fn setup() -> ([f64; 10], [DeBoorSpan<f64>; 10]) {
        // Define knot vector
        let p = 2;
        let xi = KnotVec(vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);

        // Define parametric values and corresponding span indices
        let ts = [0.0, 1.0/9.0, 2.0/9.0, 3.0/9.0, 4.0/9.0, 5.0/9.0, 6.0/9.0, 7.0/9.0, 8.0/9.0, 1.0];
        let spans = [2, 2, 2, 2, 2, 3, 3, 3, 3, 3];

        // Constructs local bases
        let local_bases = spans.map(|idx| DeBoorSpan::new(xi.clone(), p, KnotSpan(idx)));
        (ts, local_bases)
    }

    #[test]
    fn eval() {
        let (ts, bs) = setup();

        // Exact results of evaluated basis functions
        let evals_exact = [
            dvector![1.00000, 0.00000, 0.00000],
            dvector![0.60494, 0.37037, 0.02469],
            dvector![0.30864, 0.59259, 0.09877],
            dvector![0.11111, 0.66667, 0.22222],
            dvector![0.01235, 0.59259, 0.39506],
            dvector![0.39506, 0.59259, 0.01235],
            dvector![0.22222, 0.66667, 0.11111],
            dvector![0.09877, 0.59259, 0.30864],
            dvector![0.02469, 0.37037, 0.60494],
            dvector![0.00000, 0.00000, 1.00000],
        ];

        for (t, b, eval_exact) in izip!(ts, bs, evals_exact) {
            let eval = b.eval(t).transpose();
            assert_relative_eq!(eval, eval_exact, epsilon = 1e-5);
        }
    }
}