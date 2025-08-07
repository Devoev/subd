//! This example numerically solves the *diffusion-reaction* problem with homogeneous Neumann boundary conditions
//! using standard FE piecewise-linear basis functions.
//! The problem ist defined as
//! ```text
//! -div grad u + u = f   in Ω
//!      grad u · n = 0   on ∂Ω
//! ```
//! with `Ω=(0,1)²` being the unit square.

use nalgebra::{matrix, DVector, Point2, Vector1};
use nalgebra_sparse::CsrMatrix;
use std::f64::consts::PI;
use std::io;
use std::iter::zip;
use std::process::Command;
use subd::cells::quad::QuadNodes;
use subd::cg::cg;
use subd::error::l2_error::L2Norm;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::mesh::traits::MeshTopology;
use subd::operator::function::assemble_function;
use subd::operator::hodge::Hodge;
use subd::operator::laplace::Laplace;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::GaussLegendreMulti;
use subd::subd::lin_subd::basis::{PlBasisQuad, PlSpaceQuad};

/// Number of refinements for the convergence study.
const NUM_REFINE: u8 = 6;

pub fn main() -> io::Result<()> {
    // Define problem
    let u = |p: Point2<f64>| Vector1::new((p.x * PI).cos() * (p.y * PI).cos());
    let f = |p: Point2<f64>| (2.0 * PI.powi(2) + 1.0) * u(p);

    // Define mesh
    let coords_square = matrix![
            0.0, 0.0, 1.0, 1.0;
            0.0, 1.0, 1.0, 0.0
        ].transpose();

    let quads = vec![QuadNodes::from_indices(0, 1, 2, 3)];
    let mut msh = QuadVertexMesh::from_matrix(coords_square, quads);

    // Convergence study
    let mut n_dofs = vec![];
    let mut errs = vec![];
    for i in 0..NUM_REFINE {
        // Print info
        println!("Iteration {} / {NUM_REFINE}", i+1);


        // Refine mesh
        msh = msh.lin_subd().unpack();

        // Solve problem
        let (n_dof, err_l2, norm_l2) = solve(&msh, u, f);

        // Save and print
        n_dofs.push(n_dof);
        errs.push(err_l2);
        println!("  Absolute L2 error ||u - u_h||_2 = {:.7}", err_l2);
        println!("  Relative L2 error ||u - u_h||_2 / ||u||_2 = {:.5}%", err_l2 / norm_l2 * 100.0);
    }

    // Write data
    let mut writer = csv::Writer::from_path("examples/err_l2.csv")?;
    writer.write_record(&["n_dofs", "err_l2"])?;
    for data in zip(n_dofs, errs) {
        writer.serialize(data)?;
    }
    writer.flush()?;

    // Call octave plotting function
    Command::new("octave")
        .arg("error_plot.m")
        .current_dir("examples/")
        .output()?;

    Ok(())
}

/// Solves the problem with right hand side `f` and solution `u` on the given `msh` and `space`.
/// Returns the number of DOFs, the L2 error, and the relative L2 error.
fn solve(
    msh: &QuadVertexMesh<f64, 2>,
    u: impl Fn(Point2<f64>) -> Vector1<f64>,
    f: impl Fn(Point2<f64>) -> Vector1<f64>
) -> (usize, f64, f64) {
    // Define space
    let basis = PlBasisQuad(msh);
    let space = PlSpaceQuad::new(basis);

    // Define quadrature
    let ref_quad = GaussLegendreMulti::with_degrees([2, 2]);
    let quad = PullbackQuad::new(ref_quad);

    // Assemble system
    let hodge = Hodge::new(msh, &space);
    let laplace = Laplace::new(msh, &space);
    let f = assemble_function(msh, &space, quad.clone(), f);
    let m_coo = hodge.assemble(quad.clone());
    let k_coo = laplace.assemble(quad.clone());
    let m = CsrMatrix::from(&m_coo);
    let k = CsrMatrix::from(&k_coo);

    // Solve system
    let uh = cg(&(k + &m), &f, f.clone(), f.len(), 1e-15);
    let uh = space.linear_combination(uh)
        .expect("Number of coefficients doesn't match dimension of discrete space");

    // Calculate error
    let l2 = L2Norm::new(msh);
    let err_l2 = l2.error(&uh, &u, &quad);
    let norm_l2 = l2.norm(&u, &quad);

    // old way to compute the error using mass matrix
    // let u = DVector::from_iterator(msh.num_nodes(), msh.coords.iter().map(|&p| u(p).x));
    // let du = &u - &uh;
    // let err_l2 = (&m * &du).dot(&du).sqrt();
    // let norm_l2 = (&m * &u).dot(&u).sqrt();

    (space.dim(), err_l2, norm_l2)
}