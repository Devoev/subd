use crate::basis::eval::EvalBasis;
use crate::basis::traits::Basis;
use crate::bspline::cubic::CubicBspline;
use nalgebra::{Dyn, OMatrix, RealField, U1};

/// Basis functions for Catmull-Clark subdivision.
pub enum CatmullClarkBasis {
    Regular,
    Boundary,
    Corner,
    Irregular // todo
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
            CatmullClarkBasis::Irregular => {
                todo!()
            }
        }
    }
}

impl Basis for CatmullClarkBasis {
    type NumBasis = Dyn;
    type NumComponents = U1;

    fn num_basis(&self) -> usize {
        match self {
            CatmullClarkBasis::Regular => 16,
            CatmullClarkBasis::Boundary => 12,
            CatmullClarkBasis::Corner => 9,
            CatmullClarkBasis::Irregular => todo!()
        }
    }

    fn num_basis_generic(&self) -> Self::NumBasis {
        Dyn(self.num_basis())
    }

    fn num_components(&self) -> usize {
        1
    }

    fn num_components_generic(&self) -> Self::NumComponents {
        U1
    }
}

impl <T: RealField + Copy> EvalBasis<T, (T, T)> for CatmullClarkBasis {
    fn eval(&self, x: (T, T)) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        let (u, v) = x;
        match self {
            CatmullClarkBasis::Irregular => { todo!() },
            _ => {
                let (bu, bv) = self.bases();
                bv.eval(v).kronecker(&bu.eval(u))
            }
        }
    }
}