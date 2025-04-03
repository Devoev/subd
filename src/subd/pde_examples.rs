use std::sync::LazyLock;
use nalgebra::Point2;
use crate::subd::{iga, plot};
use crate::subd::examples::{COORDS_PENTAGON, COORDS_QUAD, FACES_PENTAGON, FACES_QUAD};
use crate::subd::iga::IgaFn;
use crate::subd::mesh::{LogicalMesh, QuadMesh};

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

#[test]
pub fn pde_square() {
    // Refine mesh
    let mut msh = MSH_SQUARE.clone();
    msh.lin_subd();
    msh.lin_subd();
    msh.lin_subd();

    // Define rhs function
    let f = |p: Point2<f64>| (p.x * p.y * 10.0).sin();
    let fh = IgaFn::from_fn(&msh, f);

    // Build load vector and stiffness matrix
    let num_quad = 2;
    let fi = iga::op_f_v(&msh, f, num_quad);
    let aij = iga::op_gradu_gradv(&msh, num_quad);

    // Check matrix properties
    assert!((aij.clone() - aij.transpose()).norm() < 1e-10, "Matrix is not symmetric"); // check symmetry
    assert!(aij.eigenvalues().unwrap().iter().all(|&ev| ev > 1e-8), "Matrix is not spd"); // check spd

    // Plot rhs
    let num_plot = 4;
    let f_plot = plot::plot_surf_fn(&msh, f, num_plot);
    let fh_plot = plot::plot_surf_fn_pullback(&msh, |patch, u, v| fh.eval_pullback(patch, u, v), num_plot);
    // f_plot.show_html("out/f_plot");
    fh_plot.show_html("out/fh_plot");

    // Solve system
    let ui = aij.lu().solve(&fi).expect("Could not solve linear system. Problem is not well-posed or system is ill-conditioned.");
    let uh = IgaFn::new(&msh, ui);

    let uh_plot = plot::plot_surf_fn_pullback(&msh, |patch, u, v| uh.eval_pullback(patch, u, v), num_plot);
    uh_plot.show_html("out/uh_plot");
}