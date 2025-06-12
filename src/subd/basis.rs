use crate::basis::eval::EvalBasis;
use crate::basis::traits::Basis;
use crate::bspline::cubic::{CubicBspline, Interpolating, Smooth};
use nalgebra::{Dyn, OMatrix, RealField, U1};

/// Basis functions for Catmull-Clark subdivision.
pub enum CatmullClarkBasis<T> {
    Regular(Smooth<T>, Smooth<T>),
    Boundary(Smooth<T>, Interpolating<T>),
    Corner(Interpolating<T>, Interpolating<T>),
    Irregular // todo
}

impl<T: Copy> CatmullClarkBasis<T> {
    /// Returns a pair of [`CubicBspline`] for both parametric directions.
    fn bases(&self) -> (CubicBspline<T>, CubicBspline<T>) {
        match self {
            CatmullClarkBasis::Regular(bu, bv) => {
                (CubicBspline::Smooth(*bu), CubicBspline::Smooth(*bv))
            }
            CatmullClarkBasis::Boundary(bu, bv) => {
                (CubicBspline::Smooth(*bu), CubicBspline::Interpolating(*bv))
            }
            CatmullClarkBasis::Corner(bu, bv) => {
                (CubicBspline::Interpolating(*bu), CubicBspline::Interpolating(*bv))
            }
            CatmullClarkBasis::Irregular => {
                todo!()
            }
        }
    }
}

impl<T> Basis for CatmullClarkBasis<T> {
    type NumBasis = Dyn;
    type NumComponents = U1;

    fn num_basis(&self) -> usize {
        match self {
            CatmullClarkBasis::Regular(_, _) => 16,
            CatmullClarkBasis::Boundary(_, _) => 12,
            CatmullClarkBasis::Corner(_, _) => 9,
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

impl <T: RealField + Copy> EvalBasis<T, (T, T)> for CatmullClarkBasis<T> {
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