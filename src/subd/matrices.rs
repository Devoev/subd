use nalgebra::{matrix, DMatrix, DVector, Dyn, RealField, RowDVector, SMatrix, Schur};
use std::iter::once;
use std::sync::LazyLock;

/// The `7✕8` matrix `S11` for extended catmull clark subdivision.
/// Taken from "Stam 1998" without trailing zeroes.
pub static S11: LazyLock<SMatrix<f64, 7, 8>> = LazyLock::new(|| {
    // a = 36, b = 6, c = 1, d = 24, e = 4, f = 16 (see "Stam 1998")
    // todo: implement special case n=3 (valence=3)
    matrix![
        1, 0, 0, 6, 36, 6, 0, 0;
        4, 0, 0, 4, 24, 24, 0, 0;
        6, 0, 0, 1, 6, 36, 6, 1;
        4, 0, 0, 0, 0, 24, 24, 4;
        4, 0, 0, 24, 24, 4, 0, 0;
        6, 1, 6, 36, 6, 1, 0, 0;
        4, 4, 24, 24, 0, 0, 0, 0;
   ].cast() / 64.0
});

/// The `7✕7` matrix `S12` for extended catmull clark subdivision.
/// Taken from "Stam 1998".
pub static S12: LazyLock<SMatrix<f64, 7, 7>> = LazyLock::new(|| {
    matrix![
        1, 6, 1, 0, 6, 1, 0;
        0, 4, 4, 0, 0, 0, 0;
        0, 1, 6, 1, 0, 0, 0;
        0, 0, 4, 4, 0, 0, 0;
        0, 0, 0, 0, 4, 4, 0;
        0, 0, 0, 0, 1, 6, 1;
        0, 0, 0, 0, 0, 4, 4;
    ].cast() / 64.0
});

/// The `9✕7 `matrix `S21` for extended catmull clark subdivision.
/// Taken from "Stam 1998" without trailing zeroes.
pub static S21: LazyLock<SMatrix<f64, 9, 7>> = LazyLock::new(|| {
    matrix![
        0, 0, 0, 0, 16, 0, 0;
        0, 0, 0, 0, 24, 4, 0;
        0, 0, 0, 0, 16, 16, 0;
        0, 0, 0, 0, 4, 24, 4;
        0, 0, 0, 0, 0, 16, 16;
        0, 0, 0, 4, 24, 0, 0;
        0, 0, 0, 16, 16, 0, 0;
        0, 0, 4, 24, 4, 0, 0;
        0, 0, 16, 16, 0, 0, 0
    ].cast() / 64.0
});

/// The `9✕7 `matrix `S22` for extended catmull clark subdivision. Taken from "Stam 1998".
pub static S22: LazyLock<SMatrix<f64, 9, 7>> = LazyLock::new(|| {
    matrix![
        16, 16, 0, 0, 16, 0, 0;
        4, 24, 4, 0, 4, 0, 0;
        0, 16, 16, 0, 0, 0, 0;
        0, 4, 24, 4, 0, 0, 0;
        0, 0, 16, 16, 0, 0, 0;
        4, 4, 0, 0, 24, 4, 0;
        0, 0, 0, 0, 16, 16, 0;
        0, 0, 0, 0, 4, 24, 4;
        0, 0, 0, 0, 0, 16, 16
    ].cast() / 64.0
});

/// The Schur decomposition of the `2n+8 ✕ 2n+8` extended subdivision matrix for valence `5`.
pub static EV5: LazyLock<Schur<f64, Dyn>> = LazyLock::new(|| {
    // todo: rename and change signature. Schur decomposition is not what we want here
    let (a, _) = build_extended_mats(5);
    a.schur()
});

/// Builds the `2n+1 ✕ 2n+1` subdivision matrix.
///
/// The ordering of nodes is taken from "Andersson 2016".
pub fn build_mat<T: RealField>(n: usize) -> DMatrix<T> {
    let weight = 1.0 / 16.0;
    let n_inv_squared = 1.0 / (n as f64).powi(2);

    // Assemble sub matrices
    // Faces to faces
    let ff = DMatrix::<f64>::from_diagonal_element(n, n, 4.0);

    // Edges to faces
    let mut ef = DMatrix::<f64>::from_element(n, n, 4.0);
    ef.fill_lower_triangle(0.0, 1);
    ef.fill_upper_triangle(0.0, 2);
    ef[(n - 1, 0)] = 4.0;

    // Vertex to faces
    let vf = DVector::from_element(n, 4.0);

    // Faces to edges
    let mut fe = DMatrix::<f64>::from_element(n, n, 1.0);
    fe.fill_lower_triangle(0.0, 2);
    fe.fill_upper_triangle(0.0, 1);
    fe[(0, n - 1)] = 1.0;

    // Edges to edges
    let mut ee = DMatrix::<f64>::from_element(n, n, 6.0);
    ee.fill_lower_triangle(1.0, 1);
    ee.fill_lower_triangle(0.0, 2);
    ee[(n - 1, 0)] = 1.0;
    ee.fill_upper_triangle_with_lower_triangle();

    // Vertex to edges
    let ve = DVector::from_element(n, 6.0);

    // Faces to vertex
    let fv = RowDVector::from_element(n, 4.0 * n_inv_squared);

    // Edges to vertex
    let ev = RowDVector::from_element(n, 24.0 * n_inv_squared);

    // Vertex to vertex
    let vv = (16.0 * (n as f64) - 28.0) / (n as f64);

    // Assemble total matrix
    let mut s = DMatrix::<f64>::zeros(2 * n + 1, 2 * n + 1);
    s.view_mut((0, 0), (n, n)).copy_from(&ff);
    s.view_mut((0, n), (n, n)).copy_from(&ef);
    s.view_mut((0, 2 * n), (n, 1)).copy_from(&vf);
    s.view_mut((n, 0), (n, n)).copy_from(&fe);
    s.view_mut((n, n), (n, n)).copy_from(&ee);
    s.view_mut((n, 2 * n), (n, 1)).copy_from(&ve);
    s.view_mut((2 * n, 0), (1, n)).copy_from(&fv);
    s.view_mut((2 * n, n), (1, n)).copy_from(&ev);
    s[(2 * n, 2 * n)] = vv;

    (s * weight).cast()
}

/// Reorders the columns and rows of the subdivision matrix to match the ordering of "Stam 1998".
/// The DOFs get reordered as
/// ```text
/// (F1,...,Fn,E1,...,En,V) -> (V,E1,F1,...,En,Fn)
/// ```
pub fn permute_matrix<T: RealField>(s: &DMatrix<T>) -> DMatrix<T> {
    // todo: possibly remove this method and directly build correct ordering
    let (r, _) = s.shape();
    let n = (r - 1) / 2;
    let face_edge_it = (0..n).flat_map(|i| once(i + n).chain(once(i)));
    let indices = once(2*n).chain(face_edge_it);

    // Permute columns
    let mut tmp = DMatrix::<T>::zeros(r, r);
    for (idx_new, idx_old) in indices.clone().enumerate() {
        tmp.set_column(idx_new, &s.column(idx_old));
    }

    // Permute rows
    let mut mat = DMatrix::<T>::zeros(r, r);
    for (idx_new, idx_old) in indices.enumerate() {
        mat.set_row(idx_new, &tmp.row(idx_old));
    }

    mat
}

/// Builds the extended `2n+8 ✕ 2n+8` and `2n+17 ✕ 2n+8` subdivision matrices.
///
/// The ordering of nodes is taken from "Stam 1998".
pub fn build_extended_mats<T: RealField>(n: usize) -> (DMatrix<T>, DMatrix<T>) {
    let s = permute_matrix(&build_mat(n));
    let mut a = DMatrix::<T>::zeros(2*n + 8, 2*n + 8);
    a.view_mut((0, 0), (2*n + 1, 2*n + 1)).copy_from(&s);
    a.fixed_view_mut::<7, 8>(2*n + 1, 0).copy_from(&S11.cast());
    a.fixed_view_mut::<7, 7>(2*n + 1, 2*n + 1).copy_from(&S12.cast());

    let mut a_bar = DMatrix::<T>::zeros(2*n + 17, 2*n + 8);
    a_bar.view_mut((0, 0), (2*n + 8, 2*n + 8)).copy_from(&a);
    a_bar.fixed_view_mut::<9, 7>(2*n + 8, 0).copy_from(&S21.cast());
    a_bar.fixed_view_mut::<9, 7>(2*n + 8, 2*n + 1).copy_from(&S22.cast());

    (a, a_bar)
}