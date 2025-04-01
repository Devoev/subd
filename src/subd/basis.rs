use crate::subd::catmull_clark;
use nalgebra::{matrix, one, vector, DMatrix, DVector, RealField, SVector};
use num_traits::ToPrimitive;
use std::iter::zip;

/// Evaluates the 4 cubic B-Splines at the parametric point `t`.
pub fn bspline<T: RealField + Copy>(t: T) -> SVector<T, 4> {
    let mat = matrix![
        -1.0, 3.0, -3.0, 1.0;
        3.0, -6.0, 0.0, 4.0;
        -3.0, 3.0, 3.0, 1.0;
        1.0, 0.0, 0.0, 0.0;
    ].cast::<T>() / T::from_i32(6).unwrap();

    let u_pow = vector![t.powi(3), t.powi(2), t, T::one()];
    mat * u_pow
}

/// Evaluates the derivatives of the 4 cubic B-Splines at the parametric point `t`.
pub fn bspline_deriv<T: RealField + Copy>(t: T) -> SVector<T, 4> {
    let mat = matrix![
        -1.0, 2.0, -1.0;
        3.0, -4.0, 0.0;
        -3.0, 2.0, 1.0;
        1.0, 0.0, 0.0;
    ].cast::<T>();

    let u_pow = vector![t.powi(2), t, T::one()];
    mat * u_pow
}

/// Evaluates the 3 interpolating cubic B-Splines at the parametric point `t`.
/// The basis interpolates the left boundary `t = 0`.
pub fn bspline_interpolating<T: RealField + Copy>(t: T) -> SVector<T, 3> {
    let mat = matrix![
        1.0, -6.0, 6.0;
        -2.0, 6.0, 0.0;
        1.0, 0.0, 0.0
    ].cast::<T>() / T::from_i32(6).unwrap();

    let u_pow = vector![t.powi(3), t, T::one()];
    mat * u_pow
}

/// Evaluates the regular cubic B-Spline basis at the parametric point `(u,v)`.
pub fn eval_regular<T: RealField + Copy>(u: T, v: T) -> SVector<T, 16> {
    let bu = bspline(u);
    let bv = bspline(v);
    bv.kronecker(&bu)
}

/// Evaluates the interpolating cubic B-Spline basis at the parametric point `(u,v)`.
/// The functions interpolate the boundaries `u = 0` and `v = 0` if `u_bnd` and `v_bnd` are set respectively.
pub fn eval_boundary<T: RealField + Copy>(u: T, v: T, u_bnd: bool, v_bnd: bool) -> DVector<T> {
    let eval = |t: T, bnd: bool| {
        if bnd {
            DVector::from_vec(bspline_interpolating(t).as_slice().to_vec())
        } else {
            DVector::from_vec(bspline(t).as_slice().to_vec())
        }
    };

    let bu = eval(u, u_bnd);
    let bv = eval(v, v_bnd);

    bv.kronecker(&bu)
}

/// Evaluates the basis functions of an irregular patch of valence `n` at the parametric point `(u,v)`.
pub fn eval_irregular<T: RealField + Copy + ToPrimitive>(u: T, v: T, n: usize) -> DVector<T> {
    // Transform (u,v)
    let (u, v, nsub, k) = transform(u, v);

    // Build subdivision matrices
    let (a, a_bar) = catmull_clark::build_extended_mats::<T>(n);

    // Evaluate regular basis on sub-patch
    let b = eval_regular(u, v);
    let b_perm = apply_permutation(n, b, permutation_vec(k, n));

    // Evaluate irregular basis
    a.pow((nsub - 1) as u32).transpose() * (a_bar.transpose() * b_perm)

    // todo: implement EV decomposition. Or maybe parse from Stam file?
    // let (q, t) = EV5.clone().unpack(); // todo: don't hardcode
    // let q = q.cast::<T>();
    // let lambda = Matrix::from_diagonal(&t.map_diagonal(|e| T::from_f64(e.powi((nsub - 1) as i32)).unwrap()));
    // lambda * (q.transpose() * (a_bar.transpose() * b_perm))
}

/// Transforms the given parametric values `(u,v)` to a regular sub-patch `(n,k)`.
pub fn transform<T: RealField + Copy + ToPrimitive>(mut u: T, mut v: T) -> (T, T, usize, usize) {
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
pub fn apply_permutation<T: RealField + Copy>(n: usize, b: SVector<T, 16>, p: PermutationVec) -> DVector<T> {
    let mut res = DVector::zeros(2*n + 17);
    for (bi, pi) in zip(b.iter(), p) {
        res[pi] = *bi;
    }
    res
}

/// Constructs the permutation matrix from the given permutation vector `p` and valence `n`.
fn permutation_matrix(p: PermutationVec, n: usize) -> DMatrix<usize> {
    let mut mat = DMatrix::<usize>::zeros(16, 2*n + 17);
    for (i, pi) in p.into_iter().enumerate() {
        // Set i-th row and pi-th column to 1
        mat[(i, pi)] = 1;
    }
    mat
}