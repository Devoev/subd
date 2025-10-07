//! This example numerically solves the *diffusion-reaction* problem with homogeneous Neumann boundary conditions
//! using isogeometric analysis with higher order B-Spline basis functions.
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
use subd::bspline::de_boor::MultiDeBoor;
use subd::bspline::space::BsplineSpace;
use subd::bspline::spline_geo::SplineGeo;
use subd::cells::geo::Cell;
use subd::cg::cg;
use subd::diffgeo::chart::Chart;
use subd::error::l2_error::L2Norm;
use subd::knots::knot_span::KnotSpan;
use subd::knots::knot_vec::KnotVec;
use subd::mesh::bezier::BezierMesh;
use subd::mesh::knot_mesh::KnotMesh;
use subd::mesh::cell_topology::Mesh;
use subd::operator::linear_form::assemble_function;
use subd::operator::hodge::Hodge;
use subd::operator::laplace::Laplace;
use subd::plot::plot_fn_msh;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::GaussLegendreMulti;

/// Number of refinements for the convergence study.
const NUM_REFINE: u8 = 4;

pub fn main() -> io::Result<()> {
    // Define problem
    let u = |p: Point2<f64>| Vector1::new((p.x * PI).cos() * (p.y * PI).cos());
    let f = |p: Point2<f64>| (2.0 * PI.powi(2) + 1.0) * u(p);

    // Define (lowest order) geometrical mapping
    let space_geo = BsplineSpace::new_open_uniform([2, 2], [1, 1]);
    let control_points = matrix![
        0.0, 0.0;
        1.0, 0.0;
        0.0, 1.0;
        1.0, 1.0
    ];
    let map = SplineGeo::from_matrix(control_points, &space_geo).unwrap();

    // Convergence study
    let mut n_dofs = vec![];
    let mut errs = vec![];
    for i in 0..NUM_REFINE {
        // Print info
        println!("Iteration {} / {NUM_REFINE}", i+1);

        // Construct mesh and space
        let p = 3;
        let n = 2usize.pow(i as u32) + p + 1;
        let xi = KnotVec::<f64>::new_open_uniform(n, p);
        let msh_ref = KnotMesh::from_knots([xi.clone(), xi.clone()]);
        let msh = BezierMesh::new(msh_ref, map.clone());
        let basis = MultiDeBoor::from_knots([xi.clone(), xi.clone()], [n, n], [p, p]);
        let space = BsplineSpace::new(basis);

        // Solve problem
        let (n_dof, err_l2, norm_l2) = solve(msh, space, u, f);

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

/// Solves the problem with right hand side `f` and solution `u` on the given `msh` and `space`.
/// Returns the number of DOFs, the L2 error, and the relative L2 error.
fn solve(
    msh: BezierMesh<f64, 2, 2>,
    space: BsplineSpace<f64, 2>,
    u: impl Fn(Point2<f64>) -> Vector1<f64>,
    f: impl Fn(Point2<f64>) -> Vector1<f64>
) -> (usize, f64, f64) {
    // Define quadrature
    let p = space.basis.bases[0].degree + 1;
    let ref_quad = GaussLegendreMulti::with_degrees([p, p]);
    let quad = PullbackQuad::new(ref_quad);

    // Assemble system
    let hodge = Hodge::new(&msh, &space);
    let laplace = Laplace::new(&msh, &space);
    let f = assemble_function(&msh, &space, quad.clone(), f);
    let m_coo = hodge.assemble(quad.clone());
    let k_coo = laplace.assemble(quad.clone());
    let m = CsrMatrix::from(&m_coo);
    let k = CsrMatrix::from(&k_coo);

    // Solve system
    let uh = cg(&(k + &m), &f, f.clone(), f.len(), 1e-15);
    let uh = space.linear_combination(uh)
        .expect("Number of coefficients doesn't match dimension of discrete space");

    // Calculate error
    let l2 = L2Norm::new(&msh);
    let err_l2 = l2.error(&uh, &u, &quad);
    let norm_l2 = l2.norm(&u, &quad);

    // Plot error
    let err_fn = |elem: &[KnotSpan; 2], x: [f64; 2]| -> f64 {
        let patch = msh.geo_elem(elem);
        let p = patch.geo_map().eval(x);
        u(p).x - uh.eval_on_elem(elem, x).x
    };
    plot_fn_msh(&msh, &err_fn, 10, |patch, num| {
        let [u_range, v_range] = patch.ref_elem.ranges();
        (lin_space(u_range, num).collect_vec(), lin_space(v_range, num).collect_vec())
    }).show();

    (space.dim(), err_l2, norm_l2)
}