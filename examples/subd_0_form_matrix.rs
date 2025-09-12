use nalgebra::{center, point, DMatrix, Point2};
use std::f64::consts::PI;
use approx::{relative_ne, RelativeEq};
use itertools::Itertools;
use nalgebra_sparse::CsrMatrix;
use subd::basis::space::Space;
use subd::cells::quad::QuadNodes;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::mesh::traits::MeshTopology;
use subd::operator::hodge::Hodge;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::GaussLegendreBi;
use subd::subd::catmull_clark::basis::CatmarkBasis;
use subd::subd::catmull_clark::matrices::assemble_global_mat;
use subd::subd::catmull_clark::mesh::CatmarkMesh;
use subd::subd::catmull_clark::quadrature::SubdUnitSquareQuad;
use subd::subd::catmull_clark::space::CatmarkSpace;
use subd::subd::lin_subd::basis::PlBasisQuad;

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
    let mut msh = QuadVertexMesh::new(coords, faces).lin_subd().unpack();
    let msh_catmark = CatmarkMesh::from_quad_mesh(msh.clone());
    let space_catmark = CatmarkSpace::new(CatmarkBasis(&msh_catmark));

    // Refine mesh and calculate subdivision matrix
    let num_refine = 1;
    let mut a = CsrMatrix::identity(msh.num_nodes());
    for _ in 0..num_refine {
        // Update subdivision matrix
        let (s, _, _) = assemble_global_mat(&msh);
        a = CsrMatrix::from(&s) * a;

        // Refine mesh
        msh = msh.catmark_subd().unpack();
    }

    // Define space
    let space_pl = Space::<f64, _, 2>::new(PlBasisQuad(&msh));

    // Define quadrature
    let ref_quad = GaussLegendreBi::with_degrees(2, 2);
    let quad = PullbackQuad::new(ref_quad.clone());
    let quad_catmark = PullbackQuad::new(SubdUnitSquareQuad::new(ref_quad, 10));

    // Build DEC mass matrix on refined mesh
    let mass_matrix_dec = Hodge::new(&msh, &space_pl).assemble(quad);
    let m_dec = CsrMatrix::from(&mass_matrix_dec);
    let dense = DMatrix::from(&m_dec);
    // let mass_matrix = DMatrix::from(&mass_matrix);#

    // todo: using the exact catmull-clark matrix on finer level.
    //  This should technically be exactly the matrix of the coarser level,
    //  by the refinability property, but it isn't...
    let msh = CatmarkMesh::from_quad_mesh(msh);
    let space = CatmarkSpace::new(CatmarkBasis(&msh));
    let m_dec = Hodge::new(&msh, &space).assemble(quad_catmark.clone());
    let m_dec = CsrMatrix::from(&m_dec);

    // Calculate SEC on initial mesh by using the subdivision of basis functions
    let m_sec = a.transpose() * m_dec * a;
    let mass_matrix_sec = DMatrix::from(&m_sec);

    // Calculate catmull-clark mass matrix directly on initial coarse mesh
    let m_cc = Hodge::new(&msh_catmark, &space_catmark).assemble(quad_catmark);
    let mass_matrix_cc = DMatrix::from(&m_cc);

    // println!("{:?}", mass_matrix_catmark.shape());
    // println!("{}", mass_matrix_sec);
    // println!("{}", mass_matrix_cc);
    println!("Relative error = {} %", (mass_matrix_sec - &mass_matrix_cc).norm() / mass_matrix_cc.norm() * 100.0);
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