//! This example numerically solves the *Poisson* problem with homogeneous Dirichlet boundary conditions
//! using isogeometric analysis with (regular) Catmull-Clark basis functions.
//! The problem ist defined as
//! ```text
//! -div grad u = f   in Ω
//!           u = 0   on ∂Ω
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
use subd::operator::bc::DirichletBcHom;
use subd::operator::function::assemble_function;
use subd::operator::hodge::Hodge;
use subd::operator::laplace::Laplace;
use subd::plot::plot_fn_msh;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::GaussLegendreMulti;
use subd::subd::catmull_clark::basis::CatmarkBasis;
use subd::subd::catmull_clark::mesh::CatmarkMesh;
use subd::subd::catmull_clark::patch::CatmarkPatchNodes;
use subd::subd::catmull_clark::quadrature::SubdUnitSquareQuad;
use subd::subd::catmull_clark::space::CatmarkSpace;

/// Number of refinements for the convergence study.
const NUM_REFINE: u8 = 6;

pub fn main() -> io::Result<()> {
    // Define problem
    let u = |p: Point2<f64>| Vector1::new((p.x * PI).sin() * (p.y * PI).sin());
    let f = |p: Point2<f64>| 2.0 * PI.powi(2) * u(p);

    let coords_square = matrix![
            0.0, 0.0, 1.0, 1.0;
            0.0, 1.0, 1.0, 0.0
        ].transpose();

    // Define mesh
    let quads = vec![QuadNodes::from_indices(0, 1, 2, 3)];
    let mut quad_msh = QuadVertexMesh::from_matrix(coords_square, quads);

    // Convergence study
    let mut n_dofs = vec![];
    let mut errs = vec![];
    for i in 0..NUM_REFINE {
        // Print info
        println!("Iteration {} / {NUM_REFINE}", i+1);

        // Refine and construct Catmark mesh
        quad_msh = quad_msh.catmark_subd().unpack();
        let msh = CatmarkMesh::from_quad_mesh(quad_msh.clone());

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

/// Solves the problem with right hand side `f` and solution `u` on the given `msh`.
/// Returns the number of DOFs, the L2 error, and the relative L2 error.
fn solve(msh: &CatmarkMesh<f64, 2>, u: impl Fn(Point2<f64>) -> Vector1<f64>, f: impl Fn(Point2<f64>) -> Vector1<f64>) -> (usize, f64, f64) {
    // Define space
    let basis = CatmarkBasis(msh);
    let space = CatmarkSpace::new(basis);

    // Define quadrature
    let p = 4;
    let ref_quad = GaussLegendreMulti::with_degrees([p, p]);
    let quad = PullbackQuad::new(SubdUnitSquareQuad::new(ref_quad, 1));

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

    (space.dim(), err_l2, norm_l2)
}