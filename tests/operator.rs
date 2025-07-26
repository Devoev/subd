use approx::assert_relative_eq;
use nalgebra::DMatrix;
use subd::operator::hodge::Hodge;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::GaussLegendreMulti;
use subd::subd::catmull_clark::basis::CatmarkBasis;
use subd::subd::catmull_clark::mesh::CatmarkMesh;
use subd::subd::catmull_clark::space::CatmarkSpace;
use crate::common::make_pentagon_mesh;

mod common;

#[test]
fn mass_matrix_properties() {
    let msh = make_pentagon_mesh().catmark_subd().unpack();
    let msh = CatmarkMesh::from_quad_mesh(msh);

    for p in 1..=5 {
        // Define space
        let basis = CatmarkBasis(&msh);
        let space = CatmarkSpace::new(basis);

        // Define quadrature
        let ref_quad = GaussLegendreMulti::with_degrees([p + 1, p + 1]);
        let quad = PullbackQuad::new(ref_quad);

        // Build mass matrix
        let hodge = Hodge::new(&msh, &space);
        let coo_matrix = hodge.assemble(quad);

        // Build dense matrix
        let mut mass_matrix = DMatrix::<f64>::zeros(coo_matrix.nrows(), coo_matrix.ncols());
        for (i, j, &v) in coo_matrix.triplet_iter() {
            mass_matrix[(i, j)] = v;
        }

        // Test symmetry
        assert_relative_eq!(mass_matrix, mass_matrix.transpose(), epsilon = 1e-13);

        // Test positive definiteness
        let eigenvalues = mass_matrix.eigenvalues()
            .expect("Can't compute eigenvalues of mass matrix");
        assert!(eigenvalues.iter().all(|&e| e > 0.0),
                "Mass matrix is not positive-definite. Some eigenvalues are non-positive.");
    }
}