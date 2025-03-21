use nalgebra::{matrix, one, vector, RealField, SVector};

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
pub fn eval_irregular<T: RealField + Copy>(mut u: T, mut v: T) -> SVector<T, 16> {
    // Determine number of required subdivisions
    let uf: f64 = u.to_subset().unwrap();
    let vf: f64 = u.to_subset().unwrap();
    let n = -uf.log2().min(-vf.log2()).floor() as usize;
    
    // Transform (u,v) to regular sub-patch
    let pow = T::from_i32(2_i32.pow(n as u32)).unwrap();
    let mid = T::from_f64(0.5).unwrap();
    let two = T::from_i32(2).unwrap();

    u *= pow;
    v *= pow;

    let (k, u, v) = if v < mid {
        (0, u*two - one(), v*two)
    } else if u < mid {
        (2, u*two, v*two - one())
    } else {
        (1, u*two - one(), v*two - one())
    };

    todo!("Evaluate sub-patch using regular basis functions")
}