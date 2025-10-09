//! Tests for properties of the Galerkin discretized Hodge operator, i.e *mass matrix*.
//! The mass matrix should be
//! - Symmetric: `M = Mᐪ`
//! - Positive definite: `ꟛ(M) > 0`

use crate::common::matrix_properties::{assert_is_positive_definite, assert_is_symmetric};
use crate::common::mesh_examples::{make_pentagon_mesh, make_unit_square_mesh};
use nalgebra::{matrix, DMatrix};
use std::error::Error;
use subd::bspline::de_boor::MultiDeBoor;
use subd::bspline::space::BsplineSpace;
use subd::bspline::spline_geo::SplineGeo;
use subd::knots::knot_vec::KnotVec;
use subd::mesh::bezier::BezierMesh;
use subd::mesh::knot_mesh::KnotMesh;
use subd::operator::hodge::Hodge;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::{GaussLegendreBi, GaussLegendreMulti};
use subd::subd::catmull_clark::basis::CatmarkBasis;
use subd::subd::catmull_clark::mesh::CatmarkMesh;
use subd::subd::catmull_clark::quadrature::SubdUnitSquareQuad;
use subd::subd::catmull_clark::space::CatmarkSpace;
use subd::subd::lin_subd::basis::{PlBasisQuad, PlSpaceQuad};

#[test]
fn catmark_mass_matrix_properties() -> Result<(), Box<dyn Error>> {
    let msh = make_pentagon_mesh().catmark_subd().unpack();
    let msh = CatmarkMesh::from(msh);

    // Define space
    let basis = CatmarkBasis(&msh);
    let space = CatmarkSpace::new(basis);

    // Define quadrature
    let p = 3;
    let m_max = 0; // todo: pick different value?
    let ref_quad = GaussLegendreBi::with_degrees(p, p);
    let quad = PullbackQuad::new(SubdUnitSquareQuad::new(ref_quad, m_max));

    // Build mass matrix
    let hodge = Hodge::new(&msh, &space);
    let mass_matrix = DMatrix::from(&hodge.assemble(&quad));

    // Do tests
    assert_is_symmetric(&mass_matrix, 1e-13);
    assert_is_positive_definite(&mass_matrix, 1e-13)?;
    Ok(())
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
    // let mass_matrix = DMatrix::from(&hodge.assemble(&quad));
    let mass_matrix: DMatrix<f64> = todo!("Fix bspline basis cell type");

    // Do tests
    assert_is_symmetric(&mass_matrix, 1e-13);
    assert_is_positive_definite(&mass_matrix, 1e-13)?;
    Ok(())
}

#[test]
fn pl_mass_matrix_properties() -> Result<(), Box<dyn Error>> {
    // Define quadrature
    let ref_quad = GaussLegendreBi::with_degrees(2, 2);
    let quad = PullbackQuad::new(ref_quad);

    // Mesh of unit square
    let msh = make_unit_square_mesh().lin_subd().unpack();
    let space = PlSpaceQuad::new(PlBasisQuad(&msh));

    // Do tests
    let hodge = Hodge::new(&msh, &space);
    let mass_matrix = DMatrix::from(&hodge.assemble(&quad));

    assert_is_symmetric(&mass_matrix, 1e-13);
    assert_is_positive_definite(&mass_matrix, 1e-13)?;

    // Mesh of regular pentagon
    let msh = make_pentagon_mesh().lin_subd().unpack();
    let space = PlSpaceQuad::new(PlBasisQuad(&msh));

    // Do tests
    let hodge = Hodge::new(&msh, &space);
    let mass_matrix = DMatrix::from(&hodge.assemble(&quad));

    assert_is_symmetric(&mass_matrix, 1e-13);
    assert_is_positive_definite(&mass_matrix, 1e-13)?;
    Ok(())
}