use nalgebra::{DVector, RealField};
use nalgebra_sparse::CsrMatrix;

pub fn cg<T: RealField + Copy>(a: CsrMatrix<T>, b: DVector<T>, x0: DVector<T>) {
    let r0 = b - a*x0;
    let p0 = r0;

    let k_max = 100;
    for k in 0..k_max {
        todo!()
    }
}