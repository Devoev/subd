use std::collections::{BTreeSet, HashSet};
use std::f64::consts::PI;
use std::sync::LazyLock;
use itertools::{iproduct, Itertools};
use nalgebra::{DMatrix, DVector, Point2};
use num_traits::Pow;
use crate::subd::{iga, plot};
use crate::subd::examples::{COORDS_PENTAGON, COORDS_QUAD, FACES_PENTAGON, FACES_QUAD};
use crate::subd::iga::IgaFn;
use crate::subd::mesh::{LogicalMesh, QuadMesh};
use crate::subd::patch::Patch;

/// Mesh for the regular square geometry..
static MSH_SQUARE: LazyLock<QuadMesh<f64>> = LazyLock::new(|| {
    QuadMesh {
        nodes: COORDS_QUAD.clone(),
        logical_mesh: LogicalMesh {
            faces: FACES_QUAD.clone()
        }
    }
});

/// Mesh for the irregular pentagon geometry.
static MSH_PENTAGON: LazyLock<QuadMesh<f64>> = LazyLock::new(|| {
    QuadMesh {
        nodes: COORDS_PENTAGON.clone(),
        logical_mesh: LogicalMesh {
            faces: FACES_PENTAGON.clone()
        }
    }
});

/// Test case for the diffusion-reaction problem with Neumann boundary conditions.
/// The problem ist defined as
/// ```text
/// -div grad u + u = f   in Ω
///      grad u · n = 0   on ∂Ω
/// ```
/// with `Ω=(0,1)²` being the unit square.
#[test]
pub fn dr_neu_square() {
    // Refine mesh
    let mut msh = MSH_SQUARE.clone();
    msh.lin_subd();
    msh.lin_subd();
    msh.lin_subd();

    // Define solution
    let u = |p: Point2<f64>| (p.x * PI).cos() * (p.y * PI).cos();

    // Define rhs function
    let f = |p: Point2<f64>| (2.0 * PI.powi(2) + 1.0) * u(p);
    let fh = IgaFn::from_fn(&msh, f);

    // Build load vector and stiffness matrix
    let num_quad = 2;
    let fi = iga::op_f_v(&msh, f, num_quad);
    let kij = iga::op_gradu_gradv(&msh, num_quad);
    let mij = iga::op_u_v(&msh, num_quad);
    let aij = mij + kij;

    // Check matrix properties
    assert!((aij.clone() - aij.transpose()).norm() < 1e-10, "Matrix is not symmetric"); // check symmetry
    assert!(aij.eigenvalues().unwrap().iter().all(|&ev| ev > 1e-8), "Matrix is not spd"); // check spd

    // Plot rhs
    let num_plot = 4;
    let fh_plot = plot::plot_surf_fn_pullback(&msh, |patch, u, v| fh.eval_pullback(patch, u, v), num_plot);
    // fh_plot.html.show_html("out/fh_plot.html");

    // Solve system
    let ui = aij.lu().solve(&fi).expect("Could not solve linear system. Problem is not well-posed or system is ill-conditioned.");
    let uh = IgaFn::new(&msh, ui);

    let uh_plot = plot::plot_surf_fn_pullback(&msh, |patch, u, v| uh.eval_pullback(patch, u, v), num_plot);
    // uh_plot.html.show_html("out/uh_plot.html");

    // Calculate error
    let err_fn = |patch: &Patch<f64>, t1, t2| (u(patch.eval(t1, t2)) - uh.eval_pullback(patch, t1, t2)).powi(2);
    let err_l2 = msh.integrate_pullback(err_fn, num_quad).sqrt();
    let norm_l2 = msh.integrate(|p| u(p).powi(2), num_quad).sqrt();

    dbg!(norm_l2, err_l2);
    println!("Relative L2 error ||u - u_h||_2 / ||u||_2 = {:.3}%", err_l2 / norm_l2 * 100.0);
}

/// Test case for the Poisson problem with Dirichlet boundary conditions.
/// The problem ist defined as
/// ```text
/// -div grad u = f   in Ω
///           u = 0   on ∂Ω
/// ```
/// with `Ω` being the pentagon of radius `1`.
#[test]
pub fn poisson_dir_pentagon() {
    // Refine mesh
    let mut msh = MSH_PENTAGON.clone();
    msh.lin_subd();
    msh.lin_subd();

    // Define solution
    // todo: define correct solution function
    let u = |p: Point2<f64>| (p.x * PI).cos() * (p.y * PI).cos();

    // Define rhs function
    let f = |p: Point2<f64>| (2.0 * PI.powi(2) + 1.0) * u(p);
    let fh = IgaFn::from_fn(&msh, f);

    // Define boundary condition
    let g = |p: Point2<f64>| 0.0;
    let gh = IgaFn::from_bnd_fn(&msh, u);
    let ui_bc = &gh.coeffs;

    // Plot rhs
    let num_plot = 4;
    let fh_plot = plot::plot_surf_fn_pullback(&msh, |patch, u, v| fh.eval_pullback(patch, u, v), num_plot);
    fh_plot.show_html("out/fh_plot.html");

    // Build load vector and stiffness matrix
    let num_quad = 2;
    let fi = iga::op_f_v(&msh, f, num_quad);
    let kij = iga::op_gradu_gradv(&msh, num_quad);

    // Deflate system
    let idx = (0..msh.num_nodes()).collect::<BTreeSet<_>>();
    let idx_bc = msh.boundary_nodes().collect::<BTreeSet<_>>();
    let idx_dof = idx.difference(&idx_bc).collect::<BTreeSet<_>>();

    let f_dof = DVector::from_iterator(idx_dof.len(), idx_dof.iter().map(|&&i| fi[i]));
    let k_dof_dof = DMatrix::from_iterator(idx_dof.len(), idx_dof.len(), iproduct!(idx_dof.iter(), idx_dof.iter())
        .map(|(&&i, &&j)| kij[(i, j)])
    );
    let k_dof_bc = DMatrix::from_iterator(idx_dof.len(), idx_bc.len(), iproduct!(idx_dof.iter(), idx_bc.iter())
        .map(|(&&i, &j)| kij[(i, j)])
    );

    let fi = f_dof - k_dof_bc * ui_bc; // todo: fix using mass matrix, because splines are not interpolatory
    let kij = k_dof_dof;

    // Solve system
    let mut ui = DVector::zeros(msh.num_nodes());
    let ui_dof = kij.lu().solve(&fi).expect("Could not solve linear system. Problem is not well-posed or system is ill-conditioned.");


    for (i_local, &&i) in idx_dof.iter().enumerate() {
        ui[i] = ui_dof[i_local];
    }
    for (i_local, &i) in idx_bc.iter().enumerate() {
        ui[i] = ui_bc[i_local];
    }

    let uh = IgaFn::new(&msh, ui);

    let uh_plot = plot::plot_surf_fn_pullback(&msh, |patch, u, v| uh.eval_pullback(patch, u, v), num_plot);
    uh_plot.show_html("out/uh_plot.html");

    // todo: calculate l2 error
}