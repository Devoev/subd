//! This example numerically solves the *Poisson* problem with Dirichlet boundary conditions
//! using isogeometric analysis with axisymmetric Catmull-Clark basis functions.
//! The problem ist defined as
//! ```text
//! -div grad u = 0   in Ω
//!           u = g   on ∂Ω
//! ```
//! with `Ω` being a cylinder of radius `1`.
//! The discretization is performed on a 2D axisymmetric slice of the cylinder.

use std::f64::consts::PI;
use nalgebra::{point, Point2, Vector1};
use nalgebra_sparse::CsrMatrix;
use num_traits::Zero;
use subd::cells::quad::QuadNodes;
use subd::cg::cg;
use subd::error::l2_error::L2Norm;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::operator::bc::DirichletBcHom;
use subd::operator::laplace::Laplace;
use subd::operator::linear_form::LinearForm;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::GaussLegendreBi;
use subd::subd::catmull_clark::basis::CatmarkBasis;
use subd::subd::catmull_clark::mesh::CatmarkMesh;
use subd::subd::catmull_clark::quadrature::SubdUnitSquareQuad;
use subd::subd::catmull_clark::space::CatmarkSpace;

fn main() {
    // Define problem
    // fixme: solution and right hand side are incorrect
    let u = |p: Point2<f64>| Vector1::new((p.x * PI).sin() * (p.y * PI).sin());
    let g = |p: Point2<f64>| 2.0 * PI.powi(2) * u(p);

    let coords_square = vec![
        point![0.0, 0.0],
        point![1.0, 0.0],
        point![1.0, 1.0],
        point![0.0, 1.0]
    ];

    // Define mesh
    let quads = vec![QuadNodes::new(0, 1, 2, 3)];
    let mut quad_msh = QuadVertexMesh::new(coords_square, quads);
    quad_msh = quad_msh.catmark_subd().catmark_subd().unpack();
    let msh = CatmarkMesh::from(quad_msh);

    // Solve problem
    let (n_dof, err_l2, norm_l2) = solve(&msh, u, g);

    dbg!(n_dof, err_l2, norm_l2);
}

/// Solves the problem with the boundary data `g` and solution `u` on the given `msh`.
/// Returns the number of DOFs, the L2 error, and the relative L2 error.
fn solve(msh: &CatmarkMesh<f64, 2>, u: impl Fn(Point2<f64>) -> Vector1<f64>, g: impl Fn(Point2<f64>) -> Vector1<f64>) -> (usize, f64, f64) {
    // Define space
    let basis = CatmarkBasis(msh);
    let space = CatmarkSpace::new(basis);

    // Define quadrature
    let p = 4;
    let ref_quad = GaussLegendreBi::with_degrees(p, p);
    let quad = PullbackQuad::new(SubdUnitSquareQuad::new(ref_quad, 1));

    // Assemble system
    let laplace = Laplace::new(msh, &space);
    let form = LinearForm::new(msh, &space, |_p| Vector1::zero());
    let k_coo = laplace.assemble(&quad);
    let f = form.assemble(&quad);
    let k = CsrMatrix::from(&k_coo);

    // Deflate system (homogeneous BC)
    // todo: implement non-homogeneous Dirichlet BC
    let dirichlet = DirichletBcHom::new_on_mesh(msh);
    let (k, f) = dirichlet.deflate(k, f);

    // Solve system
    let uh_dof = cg(&k, &f, f.clone(), f.len(), 1e-13);

    // Inflate system
    let uh = dirichlet.inflate(uh_dof);
    let uh = space.linear_combination(uh)
        .expect("Number of coefficients doesn't match dimension of discrete space");

    // Calculate error
    let l2 = L2Norm::new(msh);
    let err_l2 = l2.error(&uh, &u, &quad);
    let norm_l2 = l2.norm(&u, &quad);

    (space.dim(), err_l2, norm_l2)
}