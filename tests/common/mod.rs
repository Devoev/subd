use std::f64::consts::PI;
use nalgebra::{center, point, Point2};
use subd::cells::quad::QuadNodes;
use subd::mesh::face_vertex::QuadVertexMesh;

/// Constructs the center and corner points of a regular `n`-gon of radius `r`.
pub fn make_regular_polygon_geo(r: f64, n: usize) -> Vec<Point2<f64>> {
    // Angle between segments
    let phi = 2.0*PI / n as f64;

    let mut coords = vec![point![0.0, 0.0]];
    for i in 0..n {
        let phi_i = phi * i as f64;
        let phi_j = phi * (i + 1) as f64;
        let pi = point![r * phi_i.cos(), r * phi_i.sin()];
        let pj = point![r * phi_j.cos(), r * phi_j.sin()];
        coords.push(pi);
        coords.push(center(&pi, &pj));
    }
    coords
}

/// Constructs a lowest-order quad mesh for the regular pentagon of radius `1`.
pub fn make_pentagon_mesh() -> QuadVertexMesh<f64, 2> {
    let coords = make_regular_polygon_geo(1.0, 5);
    let faces = vec![
        QuadNodes::from_indices(0, 10, 1, 2),
        QuadNodes::from_indices(0, 2, 3, 4),
        QuadNodes::from_indices(0, 4, 5, 6),
        QuadNodes::from_indices(0, 6, 7, 8),
        QuadNodes::from_indices(0, 8, 9, 10),
    ];
    QuadVertexMesh::new(coords, faces)
}