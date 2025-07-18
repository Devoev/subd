//! This example numerically solves the *diffusion-reaction* problem with homogeneous Neumann boundary conditions
//! using isogeometric analysis with (regular) Catmull-Clark basis functions.
//! The problem ist defined as
//! ```text
//! -div grad u + u = f   in Ω
//!      grad u · n = 0   on ∂Ω
//! ```
//! with `Ω=(0,1)²` being the unit square.

use std::f64::consts::PI;
use nalgebra::{matrix, DVector, Point2, Vector1};
use nalgebra_sparse::CsrMatrix;
use subd::cells::quad::QuadTopo;
use subd::cg::cg;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::mesh::traits::MeshTopology;
use subd::operator::function::assemble_function;
use subd::operator::hodge::Hodge;
use subd::operator::laplace::Laplace;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::GaussLegendreMulti;
use subd::subd::catmull_clark::basis::CatmarkBasis;
use subd::subd::catmull_clark::mesh::CatmarkMesh;
use subd::subd::catmull_clark::space::CatmarkSpace;

pub fn main() {
    // Define problem
    let u = |p: Point2<f64>| Vector1::new((p.x * PI).cos() * (p.y * PI).cos());
    let f = |p: Point2<f64>| (2.0 * PI.powi(2) + 1.0) * u(p);

    let coords_square = matrix![
            0.0, 0.0, 1.0, 1.0;
            0.0, 1.0, 1.0, 0.0
        ].transpose();

    // Define mesh
    let quads = vec![QuadTopo::from_indices(0, 1, 2, 3)];
    let quad_msh = QuadVertexMesh::from_matrix(coords_square, quads);
    let refined = quad_msh.lin_subd().lin_subd().unpack();
    let msh = CatmarkMesh::from_quad_mesh(refined);

    // Define space
    let basis = CatmarkBasis(&msh);
    let space = CatmarkSpace::new(basis);

    // Define quadrature
    let p = 2;
    let ref_quad = GaussLegendreMulti::with_degrees([p, p]);
    let quad = PullbackQuad::new(ref_quad);

    // Assemble system
    let f = assemble_function(&msh, &space, quad.clone(), f);
    let hodge = Hodge::new(&msh, &space);
    let laplace = Laplace::new(&msh, &space);
    let m_coo = hodge.assemble(quad.clone());
    let k_coo = laplace.assemble(quad.clone());
    let m = CsrMatrix::from(&m_coo);
    let k = CsrMatrix::from(&k_coo);

    // Solve system
    let uh = cg(&(k + &m), &f, f.clone(), f.len(), 1e-10);

    // Calculate error
    let u = DVector::from_iterator(msh.num_nodes(), msh.coords.iter().map(|&p| u(p).x));
    let du = &u - uh;
    let err_l2 = (&m * &du).dot(&du);
    let norm_l2 = (&m * &u).dot(&u);

    println!("Absolute L2 error ||u - u_h||_2 = {:.7}", err_l2);
    println!("Relative L2 error ||u - u_h||_2 / ||u||_2 = {:.5}%", err_l2 / norm_l2 * 100.0);
}