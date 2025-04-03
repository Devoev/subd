use std::f64::consts::PI;
use std::sync::LazyLock;
use nalgebra::Point2;
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
pub fn dr_neu() {
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
    let mij = iga::op_gradu_gradv(&msh, num_quad);
    let kij = iga::op_u_v(&msh, num_quad);
    let aij = mij + kij;

    // Check matrix properties
    assert!((aij.clone() - aij.transpose()).norm() < 1e-10, "Matrix is not symmetric"); // check symmetry
    assert!(aij.eigenvalues().unwrap().iter().all(|&ev| ev > 1e-8), "Matrix is not spd"); // check spd

    // Plot rhs
    let num_plot = 4;
    let fh_plot = plot::plot_surf_fn_pullback(&msh, |patch, u, v| fh.eval_pullback(patch, u, v), num_plot);
    // fh_plot.show_html("out/fh_plot");

    // Solve system
    let ui = aij.lu().solve(&fi).expect("Could not solve linear system. Problem is not well-posed or system is ill-conditioned.");
    let uh = IgaFn::new(&msh, ui);

    let uh_plot = plot::plot_surf_fn_pullback(&msh, |patch, u, v| uh.eval_pullback(patch, u, v), num_plot);
    // uh_plot.show_html("out/uh_plot");

    // Calculate error
    let err_fn = |patch: &Patch<f64>, t1, t2| (u(patch.eval(t1, t2)) - uh.eval_pullback(patch, t1, t2)).powi(2);
    let err_l2 = msh.integrate_pullback(err_fn, num_quad).sqrt();
    let norm_l2 = msh.integrate(|p| u(p).powi(2), num_quad).sqrt();

    dbg!(norm_l2, err_l2);
    println!("Relative L2 error ||u - u_h||_2 / ||u||_2 = {:.3}%", err_l2 / norm_l2 * 100.0);
}