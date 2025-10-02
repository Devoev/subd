use nalgebra::{matrix, DMatrix, RowDVector};
use nalgebra_sparse::CsrMatrix;
use rand::random_range;
use subd::basis::space::Space;
use subd::cells::quad::QuadNodes;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::mesh::cell_topology::CellTopology;
use subd::operator::hodge::Hodge;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::GaussLegendreBi;
use subd::subd::lin_subd::basis::PlBasisQuad;
use subd::subd::lin_subd::matrix::assemble_global_mat;
use subd::subd::lin_subd::refine::LinSubd;

fn main() {
    // Define geometry
    let coords_square = matrix![
            0.0, 0.0, 1.0, 1.0;
            0.0, 1.0, 1.0, 0.0
        ].transpose();

    // Define coarse mesh and space
    let quads = vec![QuadNodes::new(0, 1, 2, 3)];
    let mut msh = QuadVertexMesh::from_coords_matrix(coords_square, quads).lin_subd().unpack();
    let msh_coarse = msh.clone();
    let space_coarse = Space::<f64, _, 2>::new(PlBasisQuad(&msh_coarse));

    // Refine mesh and calculate subdivision matrix
    let num_refine = 1;
    let mut a = CsrMatrix::identity(msh.num_nodes());
    for _ in 0..num_refine {
        // Update subdivision matrix
        let (s, _, _) = assemble_global_mat(&msh);
        a = CsrMatrix::from(&s) * a;

        // Refine mesh
        // msh = msh.lin_subd().unpack();
        LinSubd::do_refine_mat(&mut msh);
    }

    // Define fine space
    let space_fine = Space::<f64, _, 2>::new(PlBasisQuad(&msh));

    // Define quadrature
    let ref_quad = GaussLegendreBi::with_degrees(2, 2);
    let quad = PullbackQuad::new(ref_quad.clone());

    // Element for single basis evaluation
    let uv_fine = (random_range(0.0..1.0), random_range(0.0..1.0));
    let uv_coarse = (uv_fine.0 / 2_i32.pow(num_refine) as f64, uv_fine.1 / 2_i32.pow(num_refine) as f64);
    let elem_fine = &msh.elems[0];
    let elem_coarse = &msh_coarse.elems[0];

    // Build DEC mass matrix and basis eval on refined mesh
    let mut b_fine = RowDVector::zeros(msh.num_nodes());
    space_fine.populate_global_on_elem(&mut b_fine, &elem_fine, uv_fine);
    let mass_matrix_dec = Hodge::new(&msh, &space_fine).assemble(quad.clone());
    let m_dec_fine = CsrMatrix::from(&mass_matrix_dec);

    // Calculate SEC on initial mesh by using the subdivision of basis functions
    let b_sec = (a.transpose() * b_fine.transpose()).transpose();
    let m_sec_coarse = a.transpose() * m_dec_fine * a;
    let mass_matrix_sec = DMatrix::from(&m_sec_coarse);

    // Calculate mass matrix and basis eval directly on initial mesh
    let mut b_coarse = RowDVector::zeros(msh_coarse.num_nodes());
    space_coarse.populate_global_on_elem(&mut b_coarse, &elem_coarse, uv_coarse);
    let m_coarse = Hodge::new(&msh_coarse, &space_coarse).assemble(quad);
    let mass_matrix = DMatrix::from(&m_coarse);

    println!("SEC mass matrix = {}", mass_matrix_sec);
    println!("Direct mass matrix = {}", mass_matrix);
    println!("Relative error = {:e} %", (mass_matrix_sec - &mass_matrix).norm() / mass_matrix.norm() * 100.0);
    println!("Relative error of basis evaluation = {:e} %", (b_sec - &b_coarse).norm() / b_coarse.norm() * 100.0);
}