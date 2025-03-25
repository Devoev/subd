use crate::subd::catmull_clark;
use crate::subd::catmull_clark::EV5;
use nalgebra::{matrix, one, vector, DVector, Matrix, RealField, SVector};
use num_traits::ToPrimitive;
use std::iter::zip;

/// Evaluates the regular cubic B-Spline basis at the parametric point `(u,v)`.
pub fn eval_regular<T: RealField + Copy>(u: T, v: T) -> SVector<T, 16> {
    let mat = matrix![
            -1.0, 3.0, -3.0, 1.0;
            3.0, -6.0, 3.0, 0.0;
            -3.0, 0.0, 3.0, 0.0;
            1.0, 4.0, 1.0, 0.0;
        ].transpose()
        .cast::<T>() / T::from_i32(6).unwrap();
    
    let u_pow = vector![u.powi(3), u.powi(2), u, T::one()];
    let v_pow = vector![v.powi(3), v.powi(2), v, T::one()];

    let bu = mat * u_pow;
    let bv = mat * v_pow;
    bv.kronecker(&bu)
}

/// Evaluates the basis functions of an irregular patch of valence `n` at the parametric point `(u,v)`.
pub fn eval_irregular<T: RealField + Copy + ToPrimitive>(u: T, v: T, n: usize) -> DVector<T> {
    // Transform (u,v)
    let (u, v, nsub, k) = transform(u, v);

    // EV decomposition
    let (_, a_bar) = catmull_clark::build_extended_mats::<T>(n);
    let (q, t) = EV5.clone().unpack(); // todo: don't hardcode
    let q = q.cast::<T>();
    let lambda = Matrix::from_diagonal(&t.map_diagonal(|e| T::from_f64(e.powi((nsub - 1) as i32)).unwrap()));

    // Evaluate regular basis on sub-patch
    let b = eval_regular(u, v);
    let b_perm = apply_permutation(n, b, permutation_vec(k, n));

    // Evaluate irregular basis
    // todo: return eigenbasis or subd basis?
    // q.clone() * (lambda * (q.transpose() * (a_bar.transpose() * b_perm)))
    lambda * (q.transpose() * (a_bar.transpose() * b_perm))
}

/// Transforms the given parametric values `(u,v)` to a regular sub-patch `(n,k)`.
pub fn transform<T: RealField + Copy + ToPrimitive>(mut u: T, mut v: T) -> (T, T, usize, usize) {
    // Determine number of required subdivisions
    let n = (-u.log2()).min(-v.log2()).ceil().to_usize().unwrap();

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