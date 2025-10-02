use crate::space::eval_basis::{EvalBasis, EvalGrad};
use crate::space::local::MeshBasis;
use crate::space::basis::BasisFunctions;
use crate::bspline::cubic::CubicBspline;
use crate::cells::traits::{Cell};
use crate::subd::catmull_clark::matrices::{build_extended_mats, EV5};
use crate::subd::catmull_clark::mesh::CatmarkMesh;
use crate::subd::catmull_clark::patch::{CatmarkPatch, CatmarkPatchNodes};
use crate::subd::patch::subd_unit_square::SubdUnitSquare;
use nalgebra::{dvector, stack, DMatrix, Dyn, Matrix, OMatrix, RealField, RowDVector, RowSVector, SMatrix, U1, U2};
use num_traits::ToPrimitive;
use std::vec;

/// Basis functions for Catmull-Clark subdivision.
#[derive(Clone, Debug)]
pub struct CatmarkBasis<'a, T: RealField, const M: usize>(pub &'a CatmarkMesh<T, M>);

impl <'a, T: RealField, const M: usize> BasisFunctions for CatmarkBasis<'a, T, M> {
    type NumBasis = Dyn;
    type NumComponents = U1;
    type Coord<_T> = (_T, _T);

    fn num_basis_generic(&self) -> Self::NumBasis {
        Dyn(self.0.num_nodes())
    }
}

impl <'a, T: RealField + Copy + ToPrimitive, const M: usize> MeshBasis<T> for CatmarkBasis<'a, T, M> {
    type Cell = &'a CatmarkPatchNodes;
    type LocalBasis = CatmarkPatchBasis;
    type GlobalIndices = vec::IntoIter<usize>;

    fn local_basis(&self, elem: &Self::Cell) -> Self::LocalBasis {
        // todo: move this to `elem` function on CellTopo or else
        let patch = CatmarkPatch::from_msh(self.0, elem);
        patch.basis()
    }

    fn global_indices(&self, elem: &Self::Cell) -> Self::GlobalIndices {
        // todo: possibly remove clone
        elem.nodes().to_vec().into_iter()
    }
}

/// Basis functions on a Catmull-Clark patch.
#[derive(Clone, Copy, Debug)]
pub enum CatmarkPatchBasis {
    /// See [`CatmarkPatchNodes::Regular`].
    Regular,

    /// See [`CatmarkPatchNodes::Boundary`].
    Boundary,

    /// See [`CatmarkPatchNodes::Corner`].
    Corner,

    /// See [`CatmarkPatchNodes::Irregular`].
    Irregular(usize) // todo: valence parameter
}

// todo: remove eval_regular_cases and eval_regular_cases_grad methods (and possibly also bases)
//  add eval_boundary(_grads) and eval_corner(_grads) method, by using a macro or similar?

impl CatmarkPatchBasis {
    // todo: move this to a separate struct
    /// Evaluates the basis functions of the tensor product basis `bu ⊗ bv` at `(u,v)`.
    fn eval_tensor_product<T: RealField + Copy>(bu: CubicBspline, bv: CubicBspline, u: T, v: T) -> RowDVector<T> {
        bv.eval(v).kronecker(&bu.eval(u))
    }

    // todo: merge with eval_tensor_product. possibly using macros?
    /// Evaluates the `16` basis functions in the regular case [`CatmarkPatchBasis::Regular`]
    /// at `(u,v)`.
    pub fn eval_regular<T: RealField + Copy>(u: T, v: T) -> RowSVector<T, 16> {
        CubicBspline::eval_smooth(v).kronecker(&CubicBspline::eval_smooth(u))
    }

    /// Evaluates the `2n+8` basis functions for the irregular case [`CatmarkPatchBasis::Irregular`]
    /// of valence `n` at `(u,v)`.
    pub fn eval_irregular<T: RealField + Copy + ToPrimitive>(u: T, v: T, n: usize) -> RowDVector<T> {
        // For u,v = 0, return // todo: what should be returned?
        if u.is_zero() && v.is_zero() {
            // fixme: this is currently hardcoded, but seems to be the limit value. why?
            return dvector![0.5, 0.08, 0.02, 0.08, 0.02, 0.08, 0.02, 0.08, 0.02, 0.08, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].cast().transpose()
        }

        // Transform (u,v)
        let (u, v, nsub, k) = SubdUnitSquare::transform(u, v);

        // Evaluate regular basis on sub-patch
        let b = CatmarkPatchBasis::eval_regular(u, v);
        let b_perm = k.permute_basis(n, b);

        // Build subdivision matrices
        let (a, a_bar) = build_extended_mats::<T>(n);

        if n == 5 {
            // Get eigenvalue decomposition
            let (v, e, v_inv) = EV5.clone();
            let e_pows = e.map_diagonal(|ev| ev.powi((nsub - 1) as i32)).cast::<T>();

            // Evaluate irregular basis
            (b_perm * a_bar) * (v.cast::<T>() * DMatrix::from_diagonal(&e_pows) * v_inv.cast())
        } else {
            // Evaluate irregular basis
            (b_perm * a_bar) * a.pow((nsub - 1) as u32)
        }
    }

    // todo: move this to a separate struct
    /// Evaluates the gradients of the basis functions of the tensor product basis `bu ⊗ bv` at `(u,v)`.
    fn eval_tensor_product_grad<T: RealField + Copy>(bu: CubicBspline, bv: CubicBspline, u: T, v: T) -> OMatrix<T, U2, Dyn> {
        let bu_du = bu.eval_grad(u);
        let bu = bu.eval(u);
        let bv_dv = bv.eval_grad(v);
        let bv = bv.eval(v);
        let b_du = bv.kronecker(&bu_du);
        let b_dv = bv_dv.kronecker(&bu);
        Matrix::from_rows(&[b_du, b_dv])
    }

    // todo: merge with eval_tensor_product_grad. possibly using macros?
    /// Evaluates the gradients of the basis functions for the regular case [`CatmarkPatchBasis::Regular`]
    /// at `(u,v)`.
    pub fn eval_regular_grad<T: RealField + Copy + ToPrimitive>(u: T, v: T) -> SMatrix<T, 2, 16> {
        let bu_du = CubicBspline::eval_smooth_deriv(u);
        let bu = CubicBspline::eval_smooth(u);
        let bv_dv = CubicBspline::eval_smooth_deriv(v);
        let bv = CubicBspline::eval_smooth(v);
        let b_du = bv.kronecker(&bu_du);
        let b_dv = bv_dv.kronecker(&bu);
        Matrix::from_rows(&[b_du, b_dv])
    }
    
    /// Evaluates the gradients of the `2n+8` basis functions 
    /// for the irregular case [`CatmarkPatchBasis::Irregular`] of valence `n` at `(u,v)`.
    pub fn eval_irregular_grad<T: RealField + Copy + ToPrimitive>(mut u: T, mut v: T, n: usize) -> OMatrix<T, U2, Dyn> {
        // For u,v = 0, return // todo: what should be returned?
        if u.is_zero() && v.is_zero() {
            // fixme: this is currently hardcoded. Calculate exact limit and return here
            u = T::from_f64(1e-8).unwrap();
            v = T::from_f64(1e-8).unwrap();
        }

        // Transform (u,v)
        let (u, v, nsub, k) = SubdUnitSquare::transform(u, v);

        // Build subdivision matrices
        let (a, a_bar) = build_extended_mats::<T>(n);

        // Calculate 2^(n-1) using left bit shifts
        let pow = T::from_i32(1 << nsub).unwrap();

        // Evaluate regular basis on sub-patch
        let b_grad = CatmarkPatchBasis::eval_regular_grad(u, v) * pow;
        let b_du = b_grad.row(0).clone_owned();
        let b_dv = b_grad.row(1).clone_owned();
        let b_du = k.permute_basis(n, b_du);
        let b_dv = k.permute_basis(n, b_dv);
        let b_grad = stack![b_du; b_dv];

        if n == 5 {
            // Get eigenvalue decomposition
            let (v, e, v_inv) = EV5.clone();
            let e_pows = e.map_diagonal(|ev| ev.powi((nsub - 1) as i32)).cast::<T>();

            // Evaluate irregular basis
            (b_grad * a_bar) * (v.cast::<T>() * DMatrix::from_diagonal(&e_pows) * v_inv.cast())
        } else {
            // Evaluate irregular basis
            (b_grad * a_bar) * a.pow((nsub - 1) as u32)
        }
    }
}

impl BasisFunctions for CatmarkPatchBasis {
    type NumBasis = Dyn;
    type NumComponents = U1;
    type Coord<T> = (T, T);

    fn num_basis_generic(&self) -> Self::NumBasis {
        match self {
            CatmarkPatchBasis::Regular => Dyn(16),
            CatmarkPatchBasis::Boundary => Dyn(12),
            CatmarkPatchBasis::Corner => Dyn(9),
            CatmarkPatchBasis::Irregular(n) => Dyn(2*n + 8)
        }
    }
}

impl <T: RealField + Copy + ToPrimitive> EvalBasis<T> for CatmarkPatchBasis {
    fn eval(&self, x: (T, T)) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        let (u, v) = x;
        match self {
            CatmarkPatchBasis::Regular => {
                Self::eval_tensor_product(CubicBspline::Smooth, CubicBspline::Smooth, u, v)
            }
            CatmarkPatchBasis::Boundary => {
                Self::eval_tensor_product(CubicBspline::Smooth, CubicBspline::Interpolating, u, v)
            }
            CatmarkPatchBasis::Corner => {
                Self::eval_tensor_product(CubicBspline::Interpolating, CubicBspline::Interpolating, u, v)
            }
            CatmarkPatchBasis::Irregular(n) => {
                Self::eval_irregular(u, v, *n)
            }
        }
    }
}

impl <T: RealField + Copy + ToPrimitive> EvalGrad<T, 2> for CatmarkPatchBasis {
    fn eval_grad(&self, x: (T, T)) -> OMatrix<T, U2, Self::NumBasis> {
        let (u, v) = x;
        match self {
            CatmarkPatchBasis::Regular => {
                Self::eval_tensor_product_grad(CubicBspline::Smooth, CubicBspline::Smooth, u, v)
            }
            CatmarkPatchBasis::Boundary => {
                Self::eval_tensor_product_grad(CubicBspline::Smooth, CubicBspline::Interpolating, u, v)
            }
            CatmarkPatchBasis::Corner => {
                Self::eval_tensor_product_grad(CubicBspline::Interpolating, CubicBspline::Interpolating, u, v)
            }
            CatmarkPatchBasis::Irregular(n) => {
                CatmarkPatchBasis::eval_irregular_grad(u, v, *n)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use itertools::izip;
    use nalgebra::dvector;
    use rand::random_range;

    #[test]
    fn eval_regular() {
        let basis = CatmarkPatchBasis::Regular;

        // Test exact values
        let uvs = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0),];
        let evals_exact = [
            dvector![
                0.027777778, 0.111111111, 0.027777778, 0.0,
                0.111111111, 0.444444444, 0.111111111, 0.0,
                0.027777778, 0.111111111, 0.027777778, 0.0,
                0.0, 0.0, 0.0, 0.0
            ],
            dvector![
                1.42222222222222e-02, 1.05111111111112e-01, 4.71111111111117e-02, 2.22222222222217e-04,
                5.68888888888887e-02, 4.20444444444447e-01, 1.88444444444447e-01, 8.88888888888870e-04,
                1.42222222222222e-02, 1.05111111111112e-01, 4.71111111111117e-02, 2.22222222222217e-04,
                0.0, 0.0, 0.0, 0.0
            ],
            dvector![
                6e-03, 8.9777777777778e-02, 6.91111111111113e-02, 1.77777777777779e-03,
                2.4e-02, 3.59111111111112e-01, 2.76444444444445e-01, 7.11111111111114e-03,
                6e-03, 8.9777777777778e-02, 6.91111111111113e-02, 1.77777777777779e-03,
                0.0, 0.0, 0.0, 0.0
            ]
        ];

        for (uv, eval_exact) in izip!(uvs, evals_exact) {
            let eval = basis.eval(uv).transpose();
            assert_abs_diff_eq!(eval, eval_exact, epsilon = 1e-8);
        }
        
        // Test if sum equals 1
        let u = random_range(0.0..=1.0);
        let v = random_range(0.0..=1.0);
        let eval_sum = basis.eval((u, v)).sum();
        assert_abs_diff_eq!(eval_sum, 1.0, epsilon = 1e-13);
    }

    #[test]
    fn eval_boundary() {
        let basis = CatmarkPatchBasis::Boundary;

        // Test if sum equals 1
        let u = random_range(0.0..=1.0);
        let v = random_range(0.0..=1.0);
        let eval_sum = basis.eval((u, v)).sum();
        assert_abs_diff_eq!(eval_sum, 1.0, epsilon = 1e-13);
    }

    #[test]
    fn eval_corner() {
        let basis = CatmarkPatchBasis::Corner;

        // Test if sum equals 1
        let u = random_range(0.0..=1.0);
        let v = random_range(0.0..=1.0);
        let eval_sum = basis.eval((u, v)).sum();
        assert_abs_diff_eq!(eval_sum, 1.0, epsilon = 1e-13);
    }

    #[test]
    fn eval_irregular() {
        let basis = CatmarkPatchBasis::Irregular(5);

        // Test if sum equals 1
        let u = random_range(0.0..=1.0);
        let v = random_range(0.0..=1.0);
        let eval_sum = basis.eval((u, v)).sum();
        assert_abs_diff_eq!(eval_sum, 1.0, epsilon = 1e-13);
    }
}