
#[cfg(test)]
mod pde_test {
    use crate::subd::cg::cg;
    use crate::subd::examples::test_ex::{COORDS_PENTAGON, COORDS_QUAD, FACES_PENTAGON, FACES_QUAD};
    use crate::subd::iga::IgaFn;
    use crate::subd::mesh::{LogicalMesh, QuadMesh};
    use crate::subd::patch::Patch;
    use crate::subd::precompute::{BasisEval, GradEval, JacobianEval, PointEval};
    use crate::subd::quad::GaussLegendrePatch;
    use crate::subd::{iga, plot};
    use itertools::{iproduct, Itertools};
    use nalgebra::{DMatrix, DVector, Point2};
    use nalgebra_sparse::CsrMatrix;
    use std::collections::BTreeSet;
    use std::f64::consts::PI;
    use std::sync::LazyLock;

    /// Mesh for the regular square geometry.
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
        // msh.lin_subd();
        // msh.lin_subd();

        // Define solution
        let u = |p: Point2<f64>| (p.x * PI).cos() * (p.y * PI).cos();

        // Define rhs function
        let f = |p: Point2<f64>| (2.0 * PI.powi(2) + 1.0) * u(p);
        let fh = IgaFn::from_fn(&msh, f);

        // Build load vector and stiffness matrix
        let num_quad = 2;
        let quad = GaussLegendrePatch::new(num_quad).unwrap();
        let b_eval = BasisEval::from_mesh(&msh, quad.clone());
        let p_eval = PointEval::from_mesh(&msh, quad.clone());
        let grad_b_eval = GradEval::from_mesh(&msh, quad.clone());
        let j_eval = JacobianEval::from_mesh(&msh, quad.clone());
        let fi = iga::op_f_v(&msh, f, &b_eval, &p_eval, &j_eval);
        let kij = CsrMatrix::from(&iga::op_gradu_gradv(&msh, &grad_b_eval, &j_eval));
        let mij = CsrMatrix::from(&iga::op_u_v(&msh, &b_eval, &j_eval));
        let aij = kij + mij;

        // Plot rhs
        let num_plot = 4;
        let fh_plot = plot::plot_surf_fn_pullback(&msh, |patch, u, v| fh.eval_pullback(patch, u, v), num_plot);
        // fh_plot.show_html("out/fh_plot.html");

        // Solve system
        let ui = cg(&aij, &fi, fi.clone(), fi.len(), 1e-10);
        let uh = IgaFn::new(&msh, ui);

        let uh_plot = plot::plot_surf_fn_pullback(&msh, |patch, u, v| uh.eval_pullback(patch, u, v), num_plot);
        uh_plot.show_html("out/uh_plot.html");

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
        // msh.lin_subd();

        // Calculate coefficients for solution
        let coords = COORDS_PENTAGON.iter().skip(1).step_by(2).collect_vec();
        let xs = coords.iter().map(|p| p.x).collect_vec();
        let ys = coords.iter().map(|p| p.y).collect_vec();
        let a = ys.iter().circular_tuple_windows().map(|(yi, yj)| yi - yj).collect_vec();
        let b = xs.iter().circular_tuple_windows().map(|(xi, xj)| xi - xj).collect_vec();
        let c = coords.iter().circular_tuple_windows().map(|(pi, pj)| pi.x * pj.y - pj.x * pi.y).collect_vec();

        // Define solution
        let eval_factor = |i: usize, x: f64, y: f64| a[i]*x + b[i]*y + c[i];
        let eval_product = |k: usize, j: usize, x: f64, y: f64| {
            (0..5).filter(|&i| i != k && i != j)
                .map(|i| eval_factor(i, x, y))
                .product::<f64>()
        };
        let eval_deriv = |coeffs: &Vec<f64>, x: f64, y: f64| {
            (0..5).cartesian_product(0..5)
                .filter(|(k, j)| k != j)
                .map(|(k, j)| eval_product(k, j, x, y) * coeffs[k] * coeffs[j])
                .sum::<f64>()
        };

        let u = |p: Point2<f64>| {
            (0..5).map(|i| eval_factor(i, p.x, p.y)).product::<f64>()
        };

        let u_dxx = |p: Point2<f64>| eval_deriv(&a, p.x, p.y);
        let u_dyy = |p: Point2<f64>| eval_deriv(&b, p.x, p.y);

        // Define rhs function
        let f = |p: Point2<f64>| -u_dxx(p) - u_dyy(p);
        let fh = IgaFn::from_fn(&msh, f);

        // Plot rhs
        let num_plot = 4;
        let fh_plot = plot::plot_surf_fn_pullback(&msh, |patch, u, v| fh.eval_pullback(patch, u, v), num_plot);
        // fh_plot.show_html("out/fh_plot.html");

        // Define boundary condition
        let g = |_: Point2<f64>| 0.0;
        let gh = IgaFn::from_bnd_fn(&msh, g);
        let ui_bc = &gh.coeffs;

        // Build load vector and stiffness matrix
        let num_quad = 2;
        let quad = GaussLegendrePatch::new(num_quad).unwrap();
        let b_eval = BasisEval::from_mesh(&msh, quad.clone());
        let p_eval = PointEval::from_mesh(&msh, quad.clone());
        let grad_b_eval = GradEval::from_mesh(&msh, quad.clone());
        let j_eval = JacobianEval::from_mesh(&msh, quad.clone());
        let fi = iga::op_f_v(&msh, f, &b_eval, &p_eval, &j_eval);
        let kij = CsrMatrix::from(&iga::op_gradu_gradv(&msh, &grad_b_eval, &j_eval));

        // Deflate system
        let idx = (0..msh.num_nodes()).collect::<BTreeSet<_>>();
        let idx_bc = msh.boundary_nodes().collect::<BTreeSet<_>>();
        let idx_dof = idx.difference(&idx_bc).collect::<BTreeSet<_>>();

        // todo: using get_entry is expensive. Implement BC differently
        let f_dof = DVector::from_iterator(idx_dof.len(), idx_dof.iter().map(|&&i| fi[i]));
        let k_dof_dof = DMatrix::from_iterator(idx_dof.len(), idx_dof.len(), iproduct!(idx_dof.iter(), idx_dof.iter())
            .map(|(&&i, &&j)| kij.get_entry(i, j).unwrap().into_value())
        );
        let k_dof_bc = DMatrix::from_iterator(idx_dof.len(), idx_bc.len(), iproduct!(idx_dof.iter(), idx_bc.iter())
            .map(|(&&i, &j)| kij.get_entry(i, j).unwrap().into_value())
        );

        let fi = f_dof - k_dof_bc * ui_bc; // todo: fix using mass matrix, because splines are not interpolatory
        let kij = k_dof_dof;

        // Solve system
        let mut ui = DVector::zeros(msh.num_nodes());
        let kij = CsrMatrix::from(&kij);
        let ui_dof = cg(&kij, &fi, fi.clone(), fi.len(), 1e-10);
        // let ui_dof = kij.lu().solve(&fi).expect("Could not solve linear system. Problem is not well-posed or system is ill-conditioned.");


        for (i_local, &&i) in idx_dof.iter().enumerate() {
            ui[i] = ui_dof[i_local];
        }
        for (i_local, &i) in idx_bc.iter().enumerate() {
            ui[i] = ui_bc[i_local];
        }

        let uh = IgaFn::new(&msh, ui);

        let uh_plot = plot::plot_surf_fn_pullback(&msh, |patch, u, v| uh.eval_pullback(patch, u, v), num_plot);
        uh_plot.show_html("out/uh_plot.html");

        // Calculate error
        let err_fn = |patch: &Patch<f64>, t1, t2| (u(patch.eval(t1, t2)) - uh.eval_pullback(patch, t1, t2)).powi(2);
        let err_l2 = msh.integrate_pullback(err_fn, num_quad).sqrt();
        let norm_l2 = msh.integrate(|p| u(p).powi(2), num_quad).sqrt();

        println!("Relative L2 error ||u - u_h||_2 / ||u||_2 = {:.3}%", err_l2 / norm_l2 * 100.0);
    }

    /// Test case for the projection problem
    /// ```text
    /// u = f   in Ω
    /// ```
    /// with `Ω` being the pentagon of radius `1`.
    #[test]
    pub fn projection_pentagon() {
        // Refine mesh
        let mut msh = MSH_PENTAGON.clone();
        msh.lin_subd();
        msh.lin_subd();
        msh.lin_subd();

        // Define solution
        let u = |p: Point2<f64>| (p.x * PI).cos() * (p.y * PI).cos();

        // Define rhs function
        let f = u;
        let fh = IgaFn::from_fn(&msh, f);

        // Plot rhs
        let num_plot = 4;
        let fh_plot = plot::plot_surf_fn_pullback(&msh, |patch, u, v| fh.eval_pullback(patch, u, v), num_plot);
        fh_plot.show_html("out/fh_plot.html");

        // Build load vector and stiffness matrix
        let num_quad = 2;
        let quad = GaussLegendrePatch::new(num_quad).unwrap();
        let b_eval = BasisEval::from_mesh(&msh, quad.clone());
        let p_eval = PointEval::from_mesh(&msh, quad.clone());
        let j_eval = JacobianEval::from_mesh(&msh, quad.clone());
        let fi = iga::op_f_v(&msh, f, &b_eval, &p_eval, &j_eval);
        let aij = CsrMatrix::from(&iga::op_u_v(&msh, &b_eval, &j_eval));

        // Solve system
        let ui = cg(&aij, &fi, fi.clone(), fi.len(), 1e-10);
        let uh = IgaFn::new(&msh, ui);

        let uh_plot = plot::plot_surf_fn_pullback(&msh, |patch, u, v| uh.eval_pullback(patch, u, v), num_plot);
        uh_plot.show_html("out/uh_plot.html");

        // Calculate error
        let err_fn = |patch: &Patch<f64>, t1, t2| (u(patch.eval(t1, t2)) - uh.eval_pullback(patch, t1, t2)).powi(2);
        let err_l2 = msh.integrate_pullback(err_fn, num_quad).sqrt();
        let norm_l2 = msh.integrate(|p| u(p).powi(2), num_quad).sqrt();

        dbg!(norm_l2, err_l2);
        println!("Relative L2 error ||u - u_h||_2 / ||u||_2 = {:.3}%", err_l2 / norm_l2 * 100.0);
    }

}