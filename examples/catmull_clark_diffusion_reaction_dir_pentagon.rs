//! This example numerically solves the *Poisson* problem with homogeneous Dirichlet boundary conditions
//! using isogeometric analysis with (regular) Catmull-Clark basis functions.
//! The problem ist defined as
//! ```text
//! -div grad u = f   in Ω
//!           u = 0   on ∂Ω
//! ```
//! with `Ω` being the pentagon of radius `1`.

use itertools::{iproduct, Itertools};
use nalgebra::{center, point, DMatrix, DVector, Point, Point2, Vector1};
use nalgebra_sparse::CsrMatrix;
use std::collections::BTreeSet;
use std::f64::consts::PI;
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
use subd::subd::lin_subd::LinSubd;

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

fn main() {
    // Define geometry
    let coords = make_geo(1.0, 5);

    // Define solution
    let coeffs = calc_coeffs(&coords);
    let u = |p: Point2<f64>| eval_product(&coeffs, p);
    let u_dxx = |p: Point2<f64>| eval_deriv(&coeffs.0, &coeffs, p.x, p.y);
    let u_dyy = |p: Point2<f64>| eval_deriv(&coeffs.1, &coeffs, p.x, p.y);
    let f = |p: Point2<f64>| Vector1::new(-u_dxx(p) - u_dyy(p));

    // Define mesh
    let faces = vec![
        QuadTopo::from_indices(0, 10, 1, 2),
        QuadTopo::from_indices(0, 2, 3, 4),
        QuadTopo::from_indices(0, 4, 5, 6),
        QuadTopo::from_indices(0, 6, 7, 8),
        QuadTopo::from_indices(0, 8, 9, 10),
    ];
    let quad_msh = QuadVertexMesh::new(coords, faces);

    let mut lin_msh = LinSubd(quad_msh.clone());
    lin_msh.refine();
    lin_msh.refine();
    let msh = CatmarkMesh::from_quad_mesh(lin_msh.0);

    // Define space
    let basis = CatmarkBasis(&msh);
    let space = CatmarkSpace::new(basis);

    // Define quadrature
    let p = 2;
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

    // Deflate system (homogeneous BC)
    let idx = (0..msh.num_nodes()).collect::<BTreeSet<_>>();
    let idx_bc = msh.boundary_nodes().map(|n| n.0).collect::<BTreeSet<_>>();
    let idx_dof = idx.difference(&idx_bc).collect::<BTreeSet<_>>();

    // todo: using get_entry is expensive. Implement BC differently
    let f_dof = DVector::from_iterator(idx_dof.len(), idx_dof.iter().map(|&&i| f[i]));
    let k_dof_dof = DMatrix::from_iterator(idx_dof.len(), idx_dof.len(), iproduct!(idx_dof.iter(), idx_dof.iter())
        .map(|(&&i, &&j)| k.get_entry(i, j).unwrap().into_value())
    );
    let k_dof_bc = DMatrix::from_iterator(idx_dof.len(), idx_bc.len(), iproduct!(idx_dof.iter(), idx_bc.iter())
        .map(|(&&i, &j)| k.get_entry(i, j).unwrap().into_value())
    );

    let f = f_dof;
    let k = CsrMatrix::from(&k_dof_dof);

    // Solve system
    let mut uh = DVector::zeros(msh.num_nodes());
    let uh_dof = cg(&k, &f, f.clone(), f.len(), 1e-10);

    // Inflate system
    for (i_local, &&i) in idx_dof.iter().enumerate() {
        uh[i] = uh_dof[i_local];
    }

    // Calculate error
    let u = DVector::from_iterator(msh.num_nodes(), msh.coords.iter().map(|&p| u(p)));
    let du = &u - uh;
    let err_l2 = (&m * &du).dot(&du).sqrt();
    let norm_l2 = (&m * &u).dot(&u).sqrt();

    println!("Absolute L2 error ||u - u_h||_2 = {:.7}", err_l2);
    println!("Relative L2 error ||u - u_h||_2 / ||u||_2 = {:.5}%", err_l2 / norm_l2 * 100.0);
}