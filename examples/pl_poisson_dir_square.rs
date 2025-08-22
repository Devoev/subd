//! This example numerically solves the *Poisson* problem with homogeneous Dirichlet boundary conditions
//! using standard FE piecewise-linear basis functions.
//! The problem ist defined as
//! ```text
//! -div grad u = f   in Ω
//!           n = 0   on ∂Ω
//! ```
//! with `Ω=(0,1)²` being the unit square.

use itertools::{izip, Itertools};
use nalgebra::{matrix, Point2, Vector1, Vector2};
use nalgebra_sparse::CsrMatrix;
use std::f64::consts::PI;
use std::io;
use std::process::Command;
use iter_num_tools::lin_space;
use subd::cells::geo::Cell;
use subd::cells::quad::QuadNodes;
use subd::cg::cg;
use subd::diffgeo::chart::Chart;
use subd::error::h1_error::H1Norm;
use subd::error::l2_error::L2Norm;
use subd::knots::knot_span::KnotSpan;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::mesh::traits::Mesh;
use subd::operator::bc::DirichletBcHom;
use subd::operator::function::assemble_function;
use subd::operator::laplace::Laplace;
use subd::plot::plot_fn_msh;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::{GaussLegendreBi, GaussLegendreMulti};
use subd::subd::lin_subd::basis::{PlBasisQuad, PlSpaceQuad};

/// Number of refinements for the convergence study.
const NUM_REFINE: u8 = 6;

pub fn main() -> io::Result<()> {
    // Define problem
    let u = |p: Point2<f64>| Vector1::new((p.x * PI).sin() * (p.y * PI).sin());
    let u_grad = |p: Point2<f64>| Vector2::new(PI * (p.x * PI).cos() * (p.y * PI).sin(), PI * (p.x * PI).sin() * (p.y * PI).cos());
    let f = |p: Point2<f64>| 2.0 * PI.powi(2) * u(p);

    // Define mesh
    let coords_square = matrix![
            0.0, 0.0, 1.0, 1.0;
            0.0, 1.0, 1.0, 0.0
        ].transpose();

    let quads = vec![QuadNodes::from_indices(0, 1, 2, 3)];
    let mut msh = QuadVertexMesh::from_matrix(coords_square, quads);

    // Convergence study
    let mut n_dofs = vec![];
    let mut errs_l2 = vec![];
    let mut errs_h1 = vec![];
    for i in 0..NUM_REFINE {
        // Print info
        println!("Iteration {} / {NUM_REFINE}", i+1);


        // Refine mesh
        msh = msh.lin_subd().unpack();

        // Solve problem
        let (n_dof, err_h1, norm_h1, err_l2, norm_l2) = solve(&msh, u, u_grad, f);

        // Save and print
        n_dofs.push(n_dof);
        errs_l2.push(err_l2);
        errs_h1.push(err_h1);
        println!("  Absolute L2 error ||u - u_h||_2 = {:.7}", err_l2);
        println!("  Relative L2 error ||u - u_h||_2 / ||u||_2 = {:.5}%", err_l2 / norm_l2 * 100.0);
        println!("  Absolute H1 error ||u - u_h||_H1 = {:.7}", err_h1);
        println!("  Relative H1 error ||u - u_h||_H1 / ||u||_H1 = {:.5}%", err_h1 / norm_h1 * 100.0);
    }

    // Write data
    let mut writer = csv::Writer::from_path("examples/errs.csv")?;
    writer.write_record(["n_dofs", "err_l2", "err_h1"])?;
    for data in izip!(n_dofs, errs_l2, errs_h1) {
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
/// Returns the number of DOFs, the H1 error and norm, and the L2 error norm.
fn solve(
    msh: &QuadVertexMesh<f64, 2>,
    u: impl Fn(Point2<f64>) -> Vector1<f64>,
    u_grad: impl Fn(Point2<f64>) -> Vector2<f64>,
    f: impl Fn(Point2<f64>) -> Vector1<f64>
) -> (usize, f64, f64, f64, f64) {
    // Define space
    let basis = PlBasisQuad(msh);
    let space = PlSpaceQuad::new(basis);

    // Define quadrature
    let ref_quad = GaussLegendreBi::with_degrees(2, 2);
    let quad = PullbackQuad::new(ref_quad);

    // Assemble system
    let laplace = Laplace::new(msh, &space);
    let f = assemble_function(msh, &space, quad.clone(), f);
    let k_coo = laplace.assemble(quad.clone());
    let k = CsrMatrix::from(&k_coo);

    // Deflate system (homogeneous BC)
    let dirichlet = DirichletBcHom::from_mesh(msh);
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

    let h1 = H1Norm::new(msh);
    let err_h1 = h1.error(&uh, &u, &u_grad, &quad);
    let norm_h1 = h1.norm(&u, &u_grad, &quad);

    // Plot error
    let err_fn = |elem: &&QuadNodes, x: (f64, f64)| -> f64 {
        let patch = msh.geo_elem(elem);
        let p = patch.geo_map().eval(x);
        // (u_grad(p) - uh.eval_grad_on_elem(elem, x)).norm_squared()
        uh.eval_grad_on_elem(elem, x).norm_squared()
    };
    plot_fn_msh(msh, &err_fn, 10, |patch, num| {
        (lin_space(0.0..=1.0, num).collect_vec(), lin_space(0.0..=1.0, num).collect_vec())
    }).show();

    (space.dim(), err_h1, norm_h1, err_l2, norm_l2)
}