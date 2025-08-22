//! This example numerically solves the *Poisson* problem with homogeneous Dirichlet boundary conditions
//! using standard FE piecewise-linear basis functions.
//! The problem ist defined as
//! ```text
//! -div grad u = f   in Ω
//!           u = 0   on ∂Ω
//! ```
//! with `Ω` being the pentagon of circumradius `1`.

use itertools::Itertools;
use nalgebra::{center, point, Point2, Vector1};
use nalgebra_sparse::CsrMatrix;
use std::f64::consts::PI;
use std::io;
use std::iter::zip;
use std::process::Command;
use iter_num_tools::lin_space;
use subd::cells::geo::Cell;
use subd::cells::quad::QuadNodes;
use subd::cg::cg;
use subd::diffgeo::chart::Chart;
use subd::error::l2_error::L2Norm;
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
const NUM_REFINE: u8 = 3;

fn main() -> io::Result<()> {
    // Define geometry
    let coords = make_geo(1.0, 5);

    // Define solution
    let coeffs = calc_coeffs(&coords);
    let u = |p: Point2<f64>| Vector1::new(eval_product(&coeffs, p));
    let u_dxx = |p: Point2<f64>| eval_deriv(&coeffs.0, &coeffs, p.x, p.y);
    let u_dyy = |p: Point2<f64>| eval_deriv(&coeffs.1, &coeffs, p.x, p.y);
    let f = |p: Point2<f64>| Vector1::new(-u_dxx(p) - u_dyy(p));

    // Define initial mesh
    let faces = vec![
        QuadNodes::from_indices(0, 10, 1, 2),
        QuadNodes::from_indices(0, 2, 3, 4),
        QuadNodes::from_indices(0, 4, 5, 6),
        QuadNodes::from_indices(0, 6, 7, 8),
        QuadNodes::from_indices(0, 8, 9, 10),
    ];
    let mut msh = QuadVertexMesh::new(coords, faces);

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

    // Print and write
    println!("Number of dofs {n_dofs:?}");
    println!("L2 error values {errs:?}");

    let mut writer = csv::Writer::from_path("examples/errs.csv")?;
    writer.write_record(["n_dofs", "err_l2"])?;
    for data in zip(n_dofs, errs) {
        writer.serialize(data)?;
    }
    writer.flush()?;

    // Call octave plotting function
    Command::new("octave")
        // .arg("--persist")
        .arg("error_plot.m")
        // .args(&["--persist", "error_plot.m"])
        .current_dir("examples/")
        .output()?;

    Ok(())
}

/// Solves the problem with right hand side `f` and solution `u` on the given `msh`.
/// Returns the number of DOFs, the L2 error, and the relative L2 error.
fn solve(msh: &QuadVertexMesh<f64, 2>, u: impl Fn(Point2<f64>) -> Vector1<f64>, f: impl Fn(Point2<f64>) -> Vector1<f64>) -> (usize, f64, f64) {
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
    let uh_dof = cg(&k, &f, f.clone(), f.len(), 1e-10);

    // Inflate system
    let uh = dirichlet.inflate(uh_dof);
    let uh = space.linear_combination(uh)
        .expect("Number of coefficients doesn't match dimension of discrete space");

    // Plot error
    let err_fn = |elem: &&QuadNodes, x: (f64, f64)| {
        let patch = msh.geo_elem(elem);
        let p = patch.geo_map().eval(x);
        (u(p).x - uh.eval_on_elem(elem, x).x).abs()
    };
    plot_fn_msh(msh, &err_fn, 2, |_, num| {
        let grid = lin_space(0.0..=1.0, num).collect_vec();
        (grid.clone(), grid)
    }).show();

    // Calculate error
    let l2 = L2Norm::new(msh);
    let err_l2 = l2.error(&uh, &u, &quad);
    let norm_l2 = l2.norm(&u, &quad);
    (dirichlet.num_dof(), err_l2, norm_l2)
}

/// Constructs the center and corner points of a regular `n`-gon of radius `r`.
fn make_geo(r: f64, n: usize) -> Vec<Point2<f64>> {
    // Angle between segments
    let phi = 2.0*PI / n as f64;

    let mut coords = vec![point![0.0, 0.0]];
    for i in 0..n {
        let phi_i = phi * i as f64;
        let phi_j = phi * (i + 1) as f64;
        let pi = point![r * phi_i.cos(), r * phi_i.sin()];
        let pj = point![r * phi_j.cos(), r * phi_j.sin()];
        coords.push(pi);
        coords.push(center(&pi, &pj));
    }
    coords
}

/// Coefficients defining the exact solution.
type Coeffs = (Vec<f64>, Vec<f64>, Vec<f64>);

/// Calculates the coefficient vectors `a`, `b` and `c` defining the exact solution,
/// given the center and corner points of the pentagon as the vector `coords`.
fn calc_coeffs(coords: &[Point2<f64>]) -> Coeffs {
    // Get x and y coordinates of corner points
    let coords = coords.iter().skip(1).step_by(2).collect_vec();
    let (xs, ys): (Vec<f64>, Vec<f64>) = coords.iter().map(|p| (p.x, p.y)).unzip();

    // Calculate coefficients
    let a = ys.iter().circular_tuple_windows().map(|(yi, yj)| yi - yj).collect_vec();
    let b = xs.iter().circular_tuple_windows().map(|(xi, xj)| xi - xj).collect_vec();
    let c = coords.iter().circular_tuple_windows().map(|(pi, pj)| pi.x * pj.y - pj.x * pi.y).collect_vec();
    (a, b, c)
}

/// Evaluates one factor of the solution.
fn eval_factor((a, b, c): &Coeffs, i: usize, x: f64, y: f64) -> f64 {
    a[i]*x + b[i]*y + c[i]
}

/// Evaluates the product of all factors in the solution.
fn eval_product(coeffs: &Coeffs, p: Point2<f64>) -> f64 {
    (0..5).map(|i| eval_factor(coeffs, i, p.x, p.y)).product::<f64>()
}

/// Evaluates one summand of the solutions derivative.
fn eval_deriv_summand(coeffs: &Coeffs, k: usize, j: usize, x: f64, y: f64) -> f64 {
    (0..5).filter(|&i| i != k && i != j)
        .map(|i| eval_factor(coeffs, i, x, y))
        .product::<f64>()
}

/// Evaluates the partial derivative of the solution.
fn eval_deriv(deriv_coeffs: &[f64], coeffs: &Coeffs, x: f64, y: f64) -> f64 {
    (0..5).cartesian_product(0..5)
        .filter(|(k, j)| k != j)
        .map(|(k, j)| eval_deriv_summand(coeffs, k, j, x, y) * deriv_coeffs[k] * deriv_coeffs[j])
        .sum::<f64>()
}