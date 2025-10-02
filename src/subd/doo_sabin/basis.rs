use crate::space::basis::Basis;
use crate::bspline::quadratic::QuadraticBspline;
use nalgebra::{Dyn, OMatrix, RealField, RowDVector, RowSVector, U1};
use crate::space::eval_basis::EvalBasis;

/// Basis functions on a Doo-Sabin patch.
pub enum DooSabinPatchBasis {
    Regular,
    Boundary,
    Corner,
    Irregular(usize) // todo: valence parameter
}

impl DooSabinPatchBasis {
    /// Evaluates the `9` basis functions in the regular case [`DooSabinPatchBasis::Regular`]
    /// at `(u,v)`.
    fn eval_regular<T: RealField + Copy>(u: T, v: T) -> RowSVector<T, 9> {
        QuadraticBspline::eval_smooth(v).kronecker(&QuadraticBspline::eval_smooth(u))
    }

    /// Evaluates the `6` basis functions in the regular case [`DooSabinPatchBasis::Boundary`]
    /// at `(u,v)`.
    fn eval_boundary<T: RealField + Copy>(u: T, v: T) -> RowSVector<T, 6> {
        QuadraticBspline::eval_interpolating(v).kronecker(&QuadraticBspline::eval_smooth(u))
    }

    /// Evaluates the `4` basis functions in the regular case [`DooSabinPatchBasis::Corner`]
    /// at `(u,v)`.
    fn eval_corner<T: RealField + Copy>(u: T, v: T) -> RowSVector<T, 4> {
        QuadraticBspline::eval_interpolating(v).kronecker(&QuadraticBspline::eval_interpolating(u))
    }
}

impl Basis for DooSabinPatchBasis {
    type NumBasis = Dyn;
    type NumComponents = U1;
    type Coord<T> = (T, T);

    fn num_basis_generic(&self) -> Self::NumBasis {
        match self {
            DooSabinPatchBasis::Regular => Dyn(9),
            DooSabinPatchBasis::Boundary => Dyn(6),
            DooSabinPatchBasis::Corner => Dyn(4),
            DooSabinPatchBasis::Irregular(n) => todo!("How many basis functions are there in the irregular case?")
        }
    }
}

impl <T: RealField + Copy> EvalBasis<T> for DooSabinPatchBasis {
    fn eval(&self, x: (T, T)) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        let (u, v) = x;
        match self {
            DooSabinPatchBasis::Regular => {
                RowDVector::from_row_slice(DooSabinPatchBasis::eval_regular(u, v).as_slice())
            }
            DooSabinPatchBasis::Boundary => {
                RowDVector::from_row_slice(DooSabinPatchBasis::eval_boundary(u, v).as_slice())
            }
            DooSabinPatchBasis::Corner => {
                RowDVector::from_row_slice(DooSabinPatchBasis::eval_corner(u, v).as_slice())
            }
            DooSabinPatchBasis::Irregular(n) => {
                todo!("Implement irregular case")
            }
        }
    }
}