use nalgebra::{matrix, DMatrix};
use nalgebra_sparse::CsrMatrix;
use subd::basis::space::Space;
use subd::cells::quad::QuadNodes;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::mesh::traits::MeshTopology;
use subd::operator::hodge::Hodge;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::GaussLegendreBi;
use subd::subd::lin_subd::basis::PlBasisQuad;
use subd::subd::lin_subd::matrix::assemble_global_mat;

fn main() {
    // Define geometry
    let coords_square = matrix![
            0.0, 0.0, 1.0, 1.0;
            0.0, 1.0, 1.0, 0.0
        ].transpose();

    // Define coarse mesh and space
    let quads = vec![QuadNodes::from_indices(0, 1, 2, 3)];
    let mut msh = QuadVertexMesh::from_matrix(coords_square, quads).lin_subd().unpack();
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
        msh = msh.lin_subd().unpack();
    }

    // Define fine space
    let space_fine = Space::<f64, _, 2>::new(PlBasisQuad(&msh));

    // Define quadrature
    let ref_quad = GaussLegendreBi::with_degrees(4, 4);
    let quad = PullbackQuad::new(ref_quad.clone());

    // Build DEC mass matrix on refined mesh
    let mass_matrix_dec = Hodge::new(&msh, &space_fine).assemble(quad.clone());
    let m_dec_fine = CsrMatrix::from(&mass_matrix_dec);

    // Calculate SEC on initial mesh by using the subdivision of basis functions
    let m_sec_coarse = a.transpose() * m_dec_fine * a;
    let mass_matrix_sec = DMatrix::from(&m_sec_coarse);

    // Calculate catmull-clark mass matrix directly on initial mesh
    let m_coarse = Hodge::new(&msh_coarse, &space_coarse).assemble(quad);
    let mass_matrix = DMatrix::from(&m_coarse);

    // println!("{:?}", mass_matrix_catmark.shape());
    println!("{}", mass_matrix_sec);
    println!("{}", mass_matrix);
    println!("{}", &mass_matrix_sec - &mass_matrix);
    println!("Relative error = {} %", (mass_matrix_sec - &mass_matrix).norm() / mass_matrix.norm() * 100.0);
}