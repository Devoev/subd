use std::iter::zip;
use nalgebra::{matrix, min, one, vector, DVector, Matrix, RealField, SVector};
use num_traits::{AsPrimitive, ToPrimitive};
use crate::subd::catmull_clark;

/// Evaluates the regular cubic B-Spline basis at the parametric point `(u,v)`.
pub fn eval_regular<T: RealField + Copy>(u: T, v: T) -> SVector<T, 16> {
    let mat = matrix![
            -1.0, 3.0, -3.0, 1.0;
            3.0, -6.0, 3.0, 0.0;
            -3.0, 0.0, 3.0, 0.0;
            1.0, 4.0, 1.0, 0.0;
        ]
        .cast::<T>()
        / T::from_i32(6).unwrap();

    let u_pow = vector![u.powi(3), u.powi(2), u, T::one()];
    let v_pow = vector![v.powi(3), v.powi(2), v, T::one()];

    let bu = mat * u_pow;
    let bv = mat * v_pow;
    bu.kronecker(&bv)
}

/// Evaluates the irregular basis functions at the parametric point `(u,v)`.
pub fn eval_irregular<T: RealField + Copy + ToPrimitive>(mut u: T, mut v: T) -> DVector<T> {
    // Determine number of required subdivisions
    // todo: floor or ceil?
    let n= (-u.log2()).min(-v.log2()).ceil().to_usize().unwrap();

    if n == 0 {
        dbg!(u, v);
        // todo: what to do in this case? Evaluate regular patch?
        return DVector::zeros(18);
    }

    // Transform (u,v) to regular sub-patch
    let mid = T::from_f64(0.5).unwrap();
    let two = T::from_i32(2).unwrap();
    let pow = two.powi((n - 1) as i32);

    u *= pow;
    v *= pow;

    let (k, u, v) = if v < mid {
        (0, u*two - one(), v*two)
    } else if u < mid {
        (2, u*two, v*two - one())
    } else {
        (1, u*two - one(), v*two - one())
    };

    // EV decomposition
    let valence = 5; // todo: move to function signature
    let (a, a_bar) = catmull_clark::build_extended_mats::<T>(valence);
    let (q, t) = a.schur().unpack();
    let lambda = Matrix::from_diagonal(&t.map_diagonal(|e| e.powi((n - 1) as i32)));

    // Evaluate regular basis on sub-patch
    let b = eval_regular(u, v);
    let b_perm = apply_permutation(valence, b, permutation_vec(k, valence));
    
    // Evaluate irregular basis
    q.clone() * (lambda * (q.transpose() * (a_bar.transpose() * b_perm)))
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