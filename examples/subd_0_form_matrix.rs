use nalgebra::{center, matrix, point, DMatrix, Point2, RowDVector};
use nalgebra_sparse::CsrMatrix;
use std::f64::consts::PI;
use rand::random_range;
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
    let coords_square = matrix![
            0.0, 1.0, 1.0, 0.0;
            0.0, 0.0, 1.0, 1.0
        ].transpose();

    // Define coarse mesh
    let quads = vec![QuadNodes::new(0, 1, 2, 3)];
    let mut msh = QuadVertexMesh::from_coords_matrix(coords_square, quads).lin_subd().lin_subd().lin_subd().unpack();
    // // Define geometry
    // let coords = make_geo(1.0, 5);
    //
    // // Define mesh
    // let faces = vec![
    //     QuadNodes::from_indices(0, 10, 1, 2),
    //     QuadNodes::from_indices(0, 2, 3, 4),
    //     QuadNodes::from_indices(0, 4, 5, 6),
    //     QuadNodes::from_indices(0, 6, 7, 8),
    //     QuadNodes::from_indices(0, 8, 9, 10),
    // ];
    // let mut msh = QuadVertexMesh::new(coords, faces).lin_subd().unpack();
    let msh_catmark_coarse = CatmarkMesh::from(msh.clone());
    let space_catmark_coarse = CatmarkSpace::new(CatmarkBasis(&msh_catmark_coarse));

    // Refine mesh and calculate subdivision matrix
    let num_refine = 3;
    let mut a = CsrMatrix::identity(msh.num_nodes());
    for _ in 0..num_refine {
        // Update subdivision matrix
        let (s, _, _) = assemble_global_mat(&msh);
        a = CsrMatrix::from(&s) * a;

        // Refine mesh
        msh = msh.catmark_subd().unpack();
        // msh = msh.lin_subd().unpack();
        // LinSubd::do_refine_mat(&mut msh);
    }

    // Define space
    let space_pl = Space::<f64, _, 2>::new(PlBasisQuad(&msh));

    // Define quadrature
    let ref_quad = GaussLegendreBi::with_degrees(4, 4);
    let quad = PullbackQuad::new(ref_quad.clone());
    let quad_catmark = PullbackQuad::new(SubdUnitSquareQuad::new(ref_quad, 10));

    // Build DEC mass matrix on refined mesh
    let mass_matrix_dec = Hodge::new(&msh, &space_pl).assemble(quad);
    let m_dec_fine = CsrMatrix::from(&mass_matrix_dec);

    // todo: using the exact catmull-clark matrix on finer level.
    //  This should technically be EXACTLY the matrix of the coarser level,
    //  by the refinability property, but it isn't...
    // let msh_catmark_fine = CatmarkMesh::from(msh.clone());
    // let space_catmark_fine = CatmarkSpace::new(CatmarkBasis(&msh_catmark_fine));
    // let m_dec_fine = Hodge::new(&msh_catmark_fine, &space_catmark_fine).assemble(quad_catmark.clone());
    // let m_dec_fine = CsrMatrix::from(&m_dec_fine);

    // Element for single basis evaluation
    let uv_fine = (random_range(0.0..1.0), random_range(0.0..1.0));
    let uv_coarse = (uv_fine.0 / 2_i32.pow(num_refine) as f64, uv_fine.1 / 2_i32.pow(num_refine) as f64);
    // let elem_fine = &msh_catmark_fine.elems[0];
    let elem_pl = &msh.elems[0];
    let elem_coarse = &msh_catmark_coarse.elems[0];

    // Calculate SEC on initial mesh by using the subdivision of basis functions
    let mut b_pl = RowDVector::zeros(msh.num_nodes());
    space_pl.populate_global_on_elem(&mut b_pl, &elem_pl, uv_fine);
    // let mut b_fine = RowDVector::zeros(msh_catmark_fine.num_nodes());
    // space_catmark_fine.populate_global_on_elem(&mut b_fine, &elem_fine, uv_fine);

    let m_sec_coarse = a.transpose() * m_dec_fine * &a;
    let b_sec = (a.transpose() * b_pl.transpose()).transpose();
    // let b_sec = (a.transpose() * b_fine.transpose()).transpose();
    let mass_matrix_sec = DMatrix::from(&m_sec_coarse);

    // Calculate catmull-clark mass matrix directly on initial mesh
    let mut b_coarse = RowDVector::zeros(msh_catmark_coarse.num_nodes());
    space_catmark_coarse.populate_global_on_elem(&mut b_coarse, &elem_coarse, uv_coarse);

    let m_cc_coarse = Hodge::new(&msh_catmark_coarse, &space_catmark_coarse).assemble(quad_catmark);
    let mass_matrix_cc = DMatrix::from(&m_cc_coarse);

    // println!("{:?}", mass_matrix_catmark.shape());
    // println!("SEC basis evaluation = {}", b_sec);
    // println!("Direct basis evaluation = {}", b_coarse);
    // println!("SEC mass matrix = {}", mass_matrix_sec);
    // println!("Direct mass matrix = {}", mass_matrix_cc);
    // println!("Difference = {}", &mass_matrix_sec - &mass_matrix_cc);
    println!("Relative error = {:e} %", (mass_matrix_sec - &mass_matrix_cc).norm() / mass_matrix_cc.norm() * 100.0);
    println!("Relative error of basis evaluation = {:e} %", (b_sec - &b_coarse).norm() / b_coarse.norm() * 100.0);
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