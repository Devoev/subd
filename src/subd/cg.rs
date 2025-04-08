use nalgebra::{DVector, Normed, RealField};
use nalgebra_sparse::CsrMatrix;

/// Solves the linear system `ax = b` using the CG algorithm.
pub fn cg<T: RealField + Copy>(a: &CsrMatrix<T>, b: &DVector<T>, x0: DVector<T>, k_max: usize, tol: T) -> DVector<T> {
    let mut x = x0;
    let mut r = b - a*&x;
    let mut p = r.clone();

    for k in 0..k_max {
        // Test for convergence
        let rho = r.norm_squared();
        if rho < tol {
            dbg!(rho, k, k_max);
            return x;
        }

        // Calculate new iterative solution
        let v = a * &p;
        let alpha = rho / p.dot(&v);
        x += &p * alpha;
        r -= v * alpha;

        // Update value of p
        let beta = r.norm_squared() / rho;
        p = &r + p*beta;
    };

    x
}