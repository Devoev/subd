use std::collections::HashMap;
use std::iter::once;
use itertools::{chain, Itertools};
use nalgebra::{DMatrix, DVector, MatrixXx2, Point2, RealField, RowDVector};
use crate::subd::mesh::{Edge, Face, Node, QuadMesh};

/// Builds the `2n+1 ✕ 2n+1` subdivision matrix.
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

impl <T: RealField + Copy> QuadMesh<T> {

    /// Applies the Catmull-Clark scheme to this mesh.
    pub fn catmull_clark(&mut self) {
        let mut edge_points = HashMap::<Edge, Node>::new();
        let mut face_points = HashMap::<Face, Node>::new();
        let mut nodes = Vec::<Point2<T>>::new();

        for node in 0..self.num_nodes() {
            let coords = self.catmull_clark_step(node);
            for view in coords.row_iter() {
                let point = Point2::from(view.into_owned().transpose());
            }
        }
    }

    /// Performs one subdivision of the given `node`.
    fn catmull_clark_step(&self, node: Node) -> MatrixXx2<T> {
        let n = self.valence(node);
        let ring = self.node_ring(node);
        let edge_nodes = [ring[0], ring[2], ring[4], ring[6], ring[8]];
        let face_nodes = [ring[1], ring[3], ring[5], ring[7], ring[9]];

        let rows = chain!(face_nodes, edge_nodes, once(node))
            .map(|idx| self.node(idx).coords.transpose())
            .collect_vec();

        let coords = MatrixXx2::from_rows(&rows);
        let s = build_mat::<T>(n);
        s * coords
    }

}