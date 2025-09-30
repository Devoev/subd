//! This example numerically solves the *diffusion-reaction* problem with homogeneous Neumann boundary conditions
//! using isogeometric analysis with (regular) Catmull-Clark basis functions.
//! The problem ist defined as
//! ```text
//! -div grad u + u = f   in Ω
//!      grad u · n = 0   on ∂Ω
//! ```
//! with `Ω=(0,1)²` being the unit square.

use nalgebra::{matrix, Point2, Vector1};
use nalgebra_sparse::CsrMatrix;
use std::f64::consts::PI;
use std::io;
use std::iter::zip;
use std::process::Command;
use iter_num_tools::lin_space;
use itertools::Itertools;
use subd::cells::geo::Cell;
use subd::cells::quad::QuadNodes;
use subd::cg::cg;
use subd::diffgeo::chart::Chart;
use subd::error::l2_error::L2Norm;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::mesh::traits::Mesh;
use subd::operator::function::assemble_function;
use subd::operator::hodge::Hodge;
use subd::operator::laplace::Laplace;
use subd::plot::plot_fn_msh;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::{GaussLegendreBi, GaussLegendreMulti};
use subd::subd::catmull_clark::basis::CatmarkBasis;
use subd::subd::catmull_clark::mesh::CatmarkMesh;
use subd::subd::catmull_clark::patch::CatmarkPatchNodes;
use subd::subd::catmull_clark::quadrature::SubdUnitSquareQuad;
use subd::subd::catmull_clark::space::CatmarkSpace;

/// Number of refinements for the convergence study.
const NUM_REFINE: u8 = 4;

pub fn main() -> io::Result<()> {
    // Define problem
    let u = |p: Point2<f64>| Vector1::new((p.x * PI).cos() * (p.y * PI).cos());
    let f = |p: Point2<f64>| (2.0 * PI.powi(2) + 1.0) * u(p);

    // Define alt problems
    // let u = |p: Point2<f64>| Vector1::new((2.0 * p.x * PI).cos() + (2.0 * p.y * PI).cos());
    // let f = |p: Point2<f64>| (4.0 * PI.powi(2) + 1.0) * u(p);
    //
    // let u = |p: Point2<f64>| Vector1::new(
    //     (p.x.powi(4) - 2.0*p.x.powi(3) + p.x.powi(2)) * (p.y.powi(4) - 2.0*p.y.powi(3) + p.y.powi(2))
    // );
    // let f = |p: Point2<f64>| {
    //     let a = (12.0*p.x.powi(2) - 12.0*p.x + 2.0) * (p.y.powi(4) - 2.0*p.y.powi(3) + p.y.powi(2));
    //     let b = (12.0*p.y.powi(2) - 12.0*p.y + 2.0) * (p.x.powi(4) - 2.0*p.x.powi(3) + p.x.powi(2));
    //     u(p) - Vector1::new(a + b)
    // };

    let coords_square = matrix![
            0.0, 0.0, 1.0, 1.0;
            0.0, 1.0, 1.0, 0.0
        ].transpose();

    // Define mesh
    let quads = vec![QuadNodes::from_indices(0, 1, 2, 3)];
    let mut quad_msh = QuadVertexMesh::from_coords_matrix(coords_square, quads);

    // Convergence study
    let mut n_dofs = vec![];
    let mut errs = vec![];
    for i in 0..NUM_REFINE {
        // Print info
        println!("Iteration {} / {NUM_REFINE}", i+1);

        // Refine and construct Catmark mesh
        quad_msh = quad_msh.catmark_subd().unpack();
        let msh = CatmarkMesh::from(quad_msh.clone());

        // Solve problem
        let (n_dof, err_l2, norm_l2) = solve(&msh, u, f);

        // Save and print
        n_dofs.push(n_dof);
        errs.push(err_l2);
        println!("  Absolute L2 error ||u - u_h||_2 = {:.7}", err_l2);
        println!("  Relative L2 error ||u - u_h||_2 / ||u||_2 = {:.5}%", err_l2 / norm_l2 * 100.0);
    }

    // Write data
    let mut writer = csv::Writer::from_path("examples/errs.csv")?;
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

/// Solves the problem with right hand side `f` and solution `u` on the given `msh`.
/// Returns the number of DOFs, the L2 error, and the relative L2 error.
fn solve(msh: &CatmarkMesh<f64, 2>, u: impl Fn(Point2<f64>) -> Vector1<f64>, f: impl Fn(Point2<f64>) -> Vector1<f64>) -> (usize, f64, f64) {
    // Define space
    let basis = CatmarkBasis(msh);
    let space = CatmarkSpace::new(basis);

    // Define quadrature
    let p = 4;
    let ref_quad = GaussLegendreBi::with_degrees(p, p);
    let quad = PullbackQuad::new(SubdUnitSquareQuad::new(ref_quad, 1));

    // Assemble system
    let hodge = Hodge::new(msh, &space);
    let laplace = Laplace::new(msh, &space);
    let f = assemble_function(msh, &space, quad.clone(), f);
    let m_coo = hodge.assemble(quad.clone());
    let k_coo = laplace.assemble(quad.clone());
    let m = CsrMatrix::from(&m_coo);
    let k = CsrMatrix::from(&k_coo);

    // Solve system
    let uh = cg(&(k + &m), &f, f.clone(), f.len(), 1e-13);
    let uh = space.linear_combination(uh)
        .expect("Number of coefficients doesn't match dimension of discrete space");

    // Calculate error
    let l2 = L2Norm::new(msh);
    let err_l2 = l2.error(&uh, &u, &quad);
    let norm_l2 = l2.norm(&u, &quad);

    // Plot error
    let err_fn = |elem: &&CatmarkPatchNodes, x: (f64, f64)| {
        let patch = msh.geo_elem(elem);
        let p = patch.geo_map().eval(x);
        u(p).x - uh.eval_on_elem(elem, x).x
    };
    plot_fn_msh(msh, &err_fn, 10, |_, num| {
        let grid = lin_space(0.0..=1.0, num).collect_vec();
        (grid.clone(), grid)
    }).show();

    // old way to compute the error using mass matrix
    // let u = DVector::from_iterator(msh.num_nodes(), msh.coords.iter().map(|&p| u(p).x));
    // let du = &u - uh;
    // let err_l2 = (&m * &du).dot(&du).sqrt();
    // let norm_l2 = (&m * &u).dot(&u).sqrt();

    (space.dim(), err_l2, norm_l2)
}