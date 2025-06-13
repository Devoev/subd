use crate::basis::eval::EvalBasis;
use crate::basis::traits::Basis;
use crate::bspline::cubic::CubicBspline;
use nalgebra::{Dyn, OMatrix, RealField, U1};

/// Basis functions for Catmull-Clark subdivision.
pub enum CatmullClarkBasis {
    Regular,
    Boundary,
    Corner,
    Irregular(usize) // todo: valence parameter
}

impl CatmullClarkBasis {
    /// Returns a pair of [`CubicBspline`] for both parametric directions.
    fn bases(&self) -> (CubicBspline, CubicBspline) {
        match self {
            CatmullClarkBasis::Regular => {
                (CubicBspline::Smooth, CubicBspline::Smooth)
            }
            CatmullClarkBasis::Boundary => {
                (CubicBspline::Smooth, CubicBspline::Interpolating)
            }
            CatmullClarkBasis::Corner => {
                (CubicBspline::Interpolating, CubicBspline::Interpolating)
            }
            CatmullClarkBasis::Irregular(_) => {
                todo!()
            }
        }
    }
}

impl Basis for CatmullClarkBasis {
    type NumBasis = Dyn;
    type NumComponents = U1;

    fn num_basis_generic(&self) -> Self::NumBasis {
        match self {
            CatmullClarkBasis::Regular => Dyn(16),
            CatmullClarkBasis::Boundary => Dyn(12),
            CatmullClarkBasis::Corner => Dyn(9),
            CatmullClarkBasis::Irregular(n) => todo!("dependent on valence")
        }
    }
}

impl <T: RealField + Copy> EvalBasis<T, (T, T)> for CatmullClarkBasis {
    fn eval(&self, x: (T, T)) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        let (u, v) = x;
        match self {
            CatmullClarkBasis::Irregular(n) => { todo!() },
            _ => {
                let (bu, bv) = self.bases();
                bv.eval(v).kronecker(&bu.eval(u))
            }
        }
    }
}