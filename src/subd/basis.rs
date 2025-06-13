use crate::basis::eval::{EvalBasis, EvalGrad};
use crate::basis::traits::Basis;
use crate::bspline::cubic::CubicBspline;
use nalgebra::{Const, Dyn, Matrix, OMatrix, RealField, U1, U2};

/// Basis functions for Catmull-Clark subdivision.
pub enum CatmarkBasis {
    Regular,
    Boundary,
    Corner,
    Irregular(usize) // todo: valence parameter
}

impl CatmarkBasis {
    /// Returns a pair of [`CubicBspline`] for both parametric directions.
    fn bases(&self) -> (CubicBspline, CubicBspline) {
        match self {
            CatmarkBasis::Regular => {
                (CubicBspline::Smooth, CubicBspline::Smooth)
            }
            CatmarkBasis::Boundary => {
                (CubicBspline::Smooth, CubicBspline::Interpolating)
            }
            CatmarkBasis::Corner => {
                (CubicBspline::Interpolating, CubicBspline::Interpolating)
            }
            CatmarkBasis::Irregular(_) => {
                todo!()
            }
        }
    }
}

impl Basis for CatmarkBasis {
    type NumBasis = Dyn;
    type NumComponents = U1;

    fn num_basis_generic(&self) -> Self::NumBasis {
        match self {
            CatmarkBasis::Regular => Dyn(16),
            CatmarkBasis::Boundary => Dyn(12),
            CatmarkBasis::Corner => Dyn(9),
            CatmarkBasis::Irregular(n) => todo!("dependent on valence")
        }
    }
}

impl <T: RealField + Copy> EvalBasis<T, (T, T)> for CatmarkBasis {
    fn eval(&self, x: (T, T)) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        let (u, v) = x;
        match self {
            CatmarkBasis::Irregular(n) => { todo!() },
            _ => {
                let (bu, bv) = self.bases();
                bv.eval(v).kronecker(&bu.eval(u))
            }
        }
    }
}

impl <T: RealField + Copy> EvalGrad<T, (T, T), 2> for CatmarkBasis {
    fn eval_grad(&self, x: (T, T)) -> OMatrix<T, U2, Self::NumBasis> {
        let (u, v) = x;
        match self {
            CatmarkBasis::Irregular(n) => { todo!() },
            _ => {
                let (basis_u, basis_v) = self.bases();
                let bu = basis_u.eval(u);
                let bu_du = basis_u.eval_grad(u);
                let bv = basis_v.eval(v);
                let bv_dv = basis_v.eval_grad(v);
                let b_du = bv.kronecker(&bu_du);
                let b_dv = bv_dv.kronecker(&bu);
                Matrix::from_rows(&[b_du, b_dv])
            }
        }
    }
}