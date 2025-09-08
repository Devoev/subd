use std::f64::consts::PI;
use nalgebra::{center, point, DMatrix, Point2};
use nalgebra_sparse::CsrMatrix;
use subd::basis::space::Space;
use subd::cells::quad::QuadNodes;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::mesh::traits::MeshTopology;
use subd::operator::hodge::Hodge;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::GaussLegendreBi;
use subd::subd::lin_subd::edge_basis::WhitneyEdgeQuad;

fn main() {
    // Define geometry
    let coords = make_geo(1.0, 5);

    // Define mesh
    let faces = vec![
        QuadNodes::from_indices(0, 10, 1, 2),
        QuadNodes::from_indices(0, 2, 3, 4),
        QuadNodes::from_indices(0, 4, 5, 6),
        QuadNodes::from_indices(0, 6, 7, 8),
        QuadNodes::from_indices(0, 8, 9, 10),
    ];
    let mut msh = QuadVertexMesh::new(coords, faces);
    // msh = msh.lin_subd().unpack();

    // Define edge space
    let space = Space::<f64, _, 2>::new(WhitneyEdgeQuad::new(&msh));

    // Define quadrature
    let ref_quad = GaussLegendreBi::with_degrees(2, 2);
    let quad = PullbackQuad::new(ref_quad);

    // Build mass matrix
    let mass_matrix = Hodge::new(&msh, &space).assemble(quad);
    let mass_matrix = DMatrix::from(&mass_matrix);
    // todo: this is incorrect, because the pullback of the 1-forms is not considered

    println!("{}", mass_matrix);
    println!("{}", mass_matrix.rank(1e-10));
}

/// Constructs the center and corner points of a regular `n`-gon of radius `r`.
fn make_geo(r: f64, n: usize) -> Vec<Point2<f64>> {
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