//! Tests for properties of the Galerkin discretized Hodge operator, i.e *mass matrix*.
//! The mass matrix should be
//! - Symmetric: `M = Mᐪ`
//! - Positive definite: `ꟛ(M) > 0`

use crate::common::make_pentagon_mesh;
use approx::relative_eq;
use nalgebra::{matrix, DMatrix};
use nalgebra_sparse::CooMatrix;
use std::error::Error;
use subd::bspline::de_boor::MultiDeBoor;
use subd::bspline::space::BsplineSpace;
use subd::bspline::spline_geo::SplineGeo;
use subd::knots::knot_vec::KnotVec;
use subd::mesh::bezier::BezierMesh;
use subd::mesh::knot_mesh::KnotMesh;
use subd::operator::hodge::Hodge;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::GaussLegendreMulti;
use subd::subd::catmull_clark::basis::CatmarkBasis;
use subd::subd::catmull_clark::mesh::CatmarkMesh;
use subd::subd::catmull_clark::space::CatmarkSpace;

#[test]
fn catmark_mass_matrix_properties() -> Result<(), Box<dyn Error>> {
    let msh = make_pentagon_mesh().catmark_subd().unpack();
    let msh = CatmarkMesh::from_quad_mesh(msh);

    // Define space
    let basis = CatmarkBasis(&msh);
    let space = CatmarkSpace::new(basis);

    // Define quadrature
    let p = 3;
    let ref_quad = GaussLegendreMulti::with_degrees([p, p]);
    let quad = PullbackQuad::new(ref_quad);

    // Build mass matrix
    let hodge = Hodge::new(&msh, &space);
    let mass_matrix = hodge.assemble(quad);

    // Do test
    test_properties(mass_matrix)
}

#[test]
fn bspline_mass_matrix_properties() -> Result<(), Box<dyn Error>> {
    // Define (lowest order) geometrical mapping
    let space_geo = BsplineSpace::new_open_uniform([2, 2], [1, 1]);
    let control_points = matrix![
        0.0, 0.0;
        1.0, 0.0;
        0.0, 1.0;
        1.0, 1.0
    ];
    let map = SplineGeo::from_matrix(control_points, &space_geo)?;

    // Define mesh and space
    let n = 3;
    let p = 1;
    let xi = KnotVec::<f64>::new_open_uniform(n, p);
    let msh_ref = KnotMesh::from_knots([xi.clone(), xi.clone()]);
    let msh = BezierMesh::new(msh_ref, map);
    let basis = MultiDeBoor::from_knots([xi.clone(), xi.clone()], [n, n], [p, p]);
    let space = BsplineSpace::new(basis);

    // Define quadrature
    let ref_quad = GaussLegendreMulti::with_degrees([3, 3]);
    let quad = PullbackQuad::new(ref_quad);

    // Build mass matrix
    let hodge = Hodge::new(&msh, &space);
    let mass_matrix = hodge.assemble(quad);

    // Do test
    test_properties(mass_matrix)
}

/// Tests if the given `mass_matrix` is spd.
fn test_properties(mass_matrix: CooMatrix<f64>) -> Result<(), Box<dyn Error>> {
    // Build dense matrix
    let mut mat = DMatrix::zeros(mass_matrix.nrows(), mass_matrix.ncols());
    for (i, j, &v) in mass_matrix.triplet_iter() {
        mat[(i, j)] = v;
    }

    // Test symmetry
    assert!(relative_eq!(mat, mat.transpose(), epsilon = 1e-13), "Mass matrix is not symmetric");

    // Test positive definiteness
    let eigenvalues = mat.eigenvalues().ok_or("Failed to compute eigenvalues of mass matrix")?;
    assert!(eigenvalues.iter().all(|&e| e > 0.0), "Mass matrix is not positive-definite. Some eigenvalues are non-positive.");

    Ok(())
}