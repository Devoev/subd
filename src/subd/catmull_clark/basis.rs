use crate::basis::eval::{EvalBasis, EvalGrad};
use crate::basis::local::LocalBasis;
use crate::basis::traits::Basis;
use crate::bspline::cubic::CubicBspline;
use crate::cells::topo::Cell;
use crate::mesh::traits::MeshTopology;
use itertools::Itertools;
use nalgebra::{one, stack, DMatrix, Dyn, Matrix, OMatrix, RealField, RowDVector, RowSVector, SMatrix, U1, U2};
use num_traits::ToPrimitive;
use std::iter::zip;
use std::vec;
use crate::subd::catmull_clark::matrices::build_extended_mats;
use crate::subd::catmull_clark::mesh::CatmarkMesh;
use crate::subd::catmull_clark::patch::{CatmarkPatch, CatmarkPatchNodes};

/// Basis functions for Catmull-Clark subdivision.
pub struct CatmarkBasis<'a, T: RealField, const M: usize>(pub &'a CatmarkMesh<T, M>);

impl <'a, T: RealField, const M: usize> Basis for CatmarkBasis<'a, T, M> {
    type NumBasis = Dyn;
    type NumComponents = U1;

    fn num_basis_generic(&self) -> Self::NumBasis {
        Dyn(self.0.num_nodes())
    }
}

impl <'a, T: RealField + Copy + ToPrimitive, const M: usize> LocalBasis<T, (T, T)> for CatmarkBasis<'a, T, M> {
    type Elem = &'a CatmarkPatchNodes;
    type ElemBasis = CatmarkPatchBasis;
    type GlobalIndices = vec::IntoIter<usize>;

    fn find_elem(&self, x: (T, T)) -> Self::Elem {
        todo!()
    }

    fn elem_basis(&self, elem: &Self::Elem) -> Self::ElemBasis {
        // todo: move this to `elem` function on CellTopo or else
        let patch = CatmarkPatch::from_msh(self.0, elem);
        patch.basis()
    }

    fn global_indices(&self, elem: &Self::Elem) -> Self::GlobalIndices {
        // todo: possibly remove allocation
        let indices = elem.nodes().iter().map(|node| node.0).collect_vec();
        indices.into_iter()
    }
}

/// Basis functions on a Catmull-Clark patch.
pub enum CatmarkPatchBasis {
    Regular,
    Boundary,
    Corner,
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
        // Transform (u,v)
        let (u, v, nsub, k) = transform(u, v);

        // Build subdivision matrices
        let (a, a_bar) = build_extended_mats::<T>(n);

        // Evaluate regular basis on sub-patch
        let b = CatmarkPatchBasis::eval_regular(u, v);
        let b_perm = apply_permutation(n, b, permutation_vec(k, n));

        // Evaluate irregular basis
        (b_perm * a_bar) * a.pow((nsub - 1) as u32)
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
    pub fn eval_irregular_grad<T: RealField + Copy + ToPrimitive>(u: T, v: T, n: usize) -> OMatrix<T, U2, Dyn> {
        // Transform (u,v)
        let (u, v, nsub, k) = transform(u, v);

        // Build subdivision matrices
        let (a, a_bar) = build_extended_mats::<T>(n);

        // Evaluate regular basis on sub-patch
        let pow2 = T::from_i32(2).unwrap().powi(nsub as i32);
        let b_grad = CatmarkPatchBasis::eval_regular_grad(u, v) * pow2;
        let b_du = b_grad.row(0).clone_owned();
        let b_dv = b_grad.row(1).clone_owned();
        let b_du = apply_permutation(n, b_du, permutation_vec(k, n));
        let b_dv = apply_permutation(n, b_dv, permutation_vec(k, n));
        let b_grad = stack![b_du; b_dv];

        // Evaluate irregular basis
        (b_grad * a_bar) * a.pow((nsub - 1) as u32)
    }
}

impl Basis for CatmarkPatchBasis {
    type NumBasis = Dyn;
    type NumComponents = U1;

    fn num_basis_generic(&self) -> Self::NumBasis {
        match self {
            CatmarkPatchBasis::Regular => Dyn(16),
            CatmarkPatchBasis::Boundary => Dyn(12),
            CatmarkPatchBasis::Corner => Dyn(9),
            CatmarkPatchBasis::Irregular(n) => Dyn(2*n + 8)
        }
    }
}

impl <T: RealField + Copy + ToPrimitive> EvalBasis<T, (T, T)> for CatmarkPatchBasis {
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

impl <T: RealField + Copy + ToPrimitive> EvalGrad<T, (T, T), 2> for CatmarkPatchBasis {
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

// todo: move this elsewhere + refactor

/// Transforms the given parametric values `(u,v)` to a regular sub-patch `(n,k)`.
fn transform<T: RealField + Copy + ToPrimitive>(mut u: T, mut v: T) -> (T, T, usize, usize) {
    // Determine number of required subdivisions
    // For u,v = 1, set the value to 1 still
    let n = (-u.log2()).min(-v.log2()).ceil().to_usize()
        .map(|n| if n == 0 { 1 } else { n }).unwrap();

    // Transform (u,v) to regular sub-patch
    let mid = T::from_f64(0.5).unwrap();
    let two = T::from_i32(2).unwrap();
    let pow = two.powi((n - 1) as i32);

    u *= pow;
    v *= pow;

    if v < mid {
        (u*two - one(), v*two, n, 0)
    } else if u < mid {
        (u*two, v*two - one(), n, 2)
    } else {
        (u*two - one(), v*two - one(), n, 1)
    }
}

/// A permutation vector to map control points from an irregular patch to a sub-patch.
type PermutationVec = [usize; 16];

/// Builds the permutation vector mapping the control points of the irregular patch of valence `n`
/// to the control points of the `k`-th sub-patch.
pub fn permutation_vec(k: usize, n: usize) -> PermutationVec {
    let m = 2 * n;

    match k {
        0 => [
            7, 6, m + 4, m + 12,
            0, 5, m + 3, m + 11,
            3, 4, m + 2, m + 10,
            m + 6, m + 5, m + 1, m + 9
        ],
        1 => [
            0, 5, m + 3, m + 11,
            3, 4, m + 2, m + 10,
            m + 6, m + 5, m + 1, m + 9,
            m + 15, m + 14, m + 13, m + 8
        ],
        2 => [
            1, 0, 5, m + 3,
            2, 3, 4, m + 2,
            m + 7, m + 6, m + 5, m + 1,
            m + 16, m + 15, m + 14, m + 13
        ],
        _ => panic!("Value of k must be between 0 and 2.")
    }
}

/// Applies the given permutation `p` to the evaluated basis functions `b`,
/// mapping them from a regular sub-patch to the irregular patch of valence `n`.
pub fn apply_permutation<T: RealField + Copy>(n: usize, b: RowSVector<T, 16>, p: PermutationVec) -> RowDVector<T> {
    let mut res = RowDVector::zeros(2*n + 17);
    for (bi, pi) in zip(b.iter(), p) {
        res[pi] = *bi;
    }
    res
}

/// Constructs the permutation matrix from the given permutation vector `p` and valence `n`.
pub fn permutation_matrix(p: PermutationVec, n: usize) -> DMatrix<usize> {
    let mut mat = DMatrix::<usize>::zeros(16, 2*n + 17);
    for (i, pi) in p.into_iter().enumerate() {
        // Set i-th row and pi-th column to 1
        mat[(i, pi)] = 1;
    }
    mat
}