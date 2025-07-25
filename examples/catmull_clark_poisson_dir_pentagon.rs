//! This example numerically solves the *Poisson* problem with homogeneous Dirichlet boundary conditions
//! using isogeometric analysis with Catmull-Clark basis functions.
//! The problem ist defined as
//! ```text
//! -div grad u = f   in Ω
//!           u = 0   on ∂Ω
//! ```
//! with `Ω` being the pentagon of radius `1`.

use itertools::Itertools;
use nalgebra::{center, point, DVector, Point2, Vector1};
use nalgebra_sparse::CsrMatrix;
use std::f64::consts::PI;
use subd::cells::quad::QuadNodes;
use subd::cg::cg;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::mesh::traits::MeshTopology;
use subd::operator::bc::DirichletBcHom;
use subd::operator::function::assemble_function;
use subd::operator::hodge::Hodge;
use subd::operator::laplace::Laplace;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::GaussLegendreMulti;
use subd::subd::catmull_clark::basis::CatmarkBasis;
use subd::subd::catmull_clark::mesh::CatmarkMesh;
use subd::subd::catmull_clark::space::CatmarkSpace;

/// Number of refinements for the convergence study.
const NUM_REFINE: u8 = 5;

fn main() {
    // Define geometry
    let coords = make_geo(1.0, 5);

    // Define solution
    let coeffs = calc_coeffs(&coords);
    let u = |p: Point2<f64>| eval_product(&coeffs, p);
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
    let mut quad_msh = QuadVertexMesh::new(coords, faces);

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

        // Save and print error
        n_dofs.push(n_dof);
        errs.push(err_l2);
        println!("  Absolute L2 error ||u - u_h||_2 = {:.7}", err_l2);
        println!("  Relative L2 error ||u - u_h||_2 / ||u||_2 = {:.5}%", err_l2 / norm_l2 * 100.0);
    }
    
    println!("Number of dofs {n_dofs:?}");
    println!("L2 error values {errs:?}");
}

/// Solves the problem with right hand side `f` and solution `u` on the given `msh`.
/// Returns the number of DOFs, the L2 error, and the relative L2 error.
fn solve(msh: &CatmarkMesh<f64, 2>, u: impl Fn(Point2<f64>) -> f64, f: impl Fn(Point2<f64>) -> Vector1<f64>) -> (usize, f64, f64) {
    // Define space
    let basis = CatmarkBasis(msh);
    let space = CatmarkSpace::new(basis);

    // Define quadrature
    let p = 2;
    let ref_quad = GaussLegendreMulti::with_degrees([p, p]);
    let quad = PullbackQuad::new(ref_quad);

    // Assemble system
    let hodge = Hodge::new(msh, &space);
    let laplace = Laplace::new(msh, &space);
    let f = assemble_function(msh, &space, quad.clone(), f);
    let m_coo = hodge.assemble(quad.clone());
    let k_coo = laplace.assemble(quad.clone());
    let m = CsrMatrix::from(&m_coo);
    let k = CsrMatrix::from(&k_coo);

    // Deflate system (homogeneous BC)
    let dirichlet = DirichletBcHom::from_mesh(msh);
    let (k, f) = dirichlet.deflate(k, f);

    // Solve system
    let uh_dof = cg(&k, &f, f.clone(), f.len(), 1e-10);

    // Inflate system
    let uh = dirichlet.inflate(uh_dof);

    // Calculate error
    let u = DVector::from_iterator(msh.num_nodes(), msh.coords.iter().map(|&p| u(p)));
    let du = &u - uh;
    let err_l2 = (&m * &du).dot(&du).sqrt();
    let norm_l2 = (&m * &u).dot(&u).sqrt();
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