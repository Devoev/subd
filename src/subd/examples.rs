
#[cfg(test)]
pub mod test_ex {
    use crate::subd::catmull_clark::{S11, S12, S21, S22};
    use crate::subd::iga::{op_f_v, op_gradu_gradv, op_u_v, IgaFn};
    use crate::subd::mesh::{Face, LogicalMesh, QuadMesh};
    use crate::subd::patch::{NodeConnectivity, Patch};
    use crate::subd::precompute::{BasisEval, GradEval, JacobianEval, PointEval};
    use crate::subd::quad::GaussLegendrePatch;
    use crate::subd::{basis, catmull_clark, patch, plot};
    use iter_num_tools::lin_space;
    use itertools::Itertools;
    use nalgebra::{center, point, Matrix, Point2, SMatrix};
    use plotly::{Plot, Scatter};
    use std::f64::consts::PI;
    use std::sync::LazyLock;
    use std::time::Instant;

    /// Vector of coordinates in 2D.
    type Coords = Vec<Point2<f64>>;

    /// Vector of faces.
    type Faces = Vec<Face>;

    /// Rectangle coordinates.
    pub static COORDS_QUAD: LazyLock<Coords> = LazyLock::new(|| {
        vec![point![0.0, 0.0], point![1.0, 0.0], point![1.0, 1.0], point![0.0, 1.0]]
    });

    /// Rectangle faces.
    pub static FACES_QUAD: LazyLock<Faces> = LazyLock::new(|| {
        vec![[0, 1, 2, 3]]
    });

    /// Fichera coordinates.
    pub static COORDS_FICHERA: LazyLock<Coords> = LazyLock::new(|| {
        vec![
            point![0.0, 0.0], point![0.0, 0.5], point![0.0, 1.0],
            point![0.5, 0.0], point![0.5, 0.5], point![0.5, 1.0],
            point![1.0, 0.0], point![1.0, 0.5]
        ]
    });

    /// Fichera faces.
    pub static FACES_FICHERA: LazyLock<Faces> = LazyLock::new(|| {
        vec![[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6]]
    });

    /// Star coordinates.
    pub static COORDS_STAR: LazyLock<Coords> = LazyLock::new(|| {
        vec![
            point![0.0, 0.0],
            point![2.0, -1.0],
            point![3.0, 1.0],
            point![1.0, 2.0],
            point![0.0, 4.0],
            point![-1.0, 2.0],
            point![-3.0, 1.0],
            point![-2.0, -1.0],
            point![-2.0, -4.0],
            point![-0.0, -3.0],
            point![2.0, -4.0]
        ]
    });

    /// Star faces.
    pub static FACES_STAR: LazyLock<Faces> = LazyLock::new(|| {
        vec![
            [0, 1, 2, 3],
            [0, 3, 4, 5],
            [7, 0, 5, 6],
            [8, 9, 0, 7],
            [9, 10, 1, 0]
        ]
    });

    /// Pentagon coordinates.
    pub static COORDS_PENTAGON: LazyLock<Coords> = LazyLock::new(|| {
        let r = 1;
        let n = 5;
        let phi = 2.0*PI / n as f64;

        let mut coords = vec![point![0.0, 0.0]];

        for i in 0..n {
            let phi_i = phi * i as f64;
            let phi_j = phi * (i + 1) as f64;
            let pi = point![phi_i.cos(), phi_i.sin()];
            let pj = point![phi_j.cos(), phi_j.sin()];
            coords.push(pi);
            coords.push(center(&pi, &pj));
        }

        coords
    });

    /// Pentagon faces.
    pub static FACES_PENTAGON: LazyLock<Faces> = LazyLock::new(|| {
        vec![
            [0, 10, 1, 2],
            [0, 2, 3, 4],
            [0, 4, 5, 6],
            [0, 6, 7, 8],
            [0, 8, 9, 10]
        ]
    });

    /// The used quad mesh.
    static MSH: LazyLock<QuadMesh<f64>> = LazyLock::new(|| {
        QuadMesh {
            nodes: COORDS_PENTAGON.clone(),
            logical_mesh: LogicalMesh {
                faces: FACES_PENTAGON.clone()
            }
        }
    });

    #[test]
    fn msh() {
        // Refine mesh
        let mut msh = MSH.clone();
        msh.lin_subd();
        msh.lin_subd();

        // Plot of mesh
        let msh_plot = plot::plot_faces(&msh, msh.faces.iter().copied());
        msh_plot.show_html("out/msh.html");

        let msh_nodes_plot = plot::plot_nodes(&msh, 0..msh.num_nodes());
        msh_nodes_plot.show_html("out/msh_nodes.html");

        // Plot extended patch
        // for face_id in 0..=3 {
        //     let patch = msh.find_patch(msh.faces[face_id]);
        //     let patch_plot = plot_faces(&msh, patch.faces.iter().copied());
        //     patch_plot.show_html(format!("plot_{face_id}.html"));
        // }

        // Plots
        // let patch_plot = plot_nodes(&msh, patch.nodes_regular().into_iter());
        // patch_plot.show_html("patch.html");

        // let patch_irr_plot = plot_nodes(&msh, patch_irr.nodes_irregular().into_iter());
        // patch_irr_plot.show_html("patch_irr.html");

        // let patch_ext_plot = plot_nodes(&msh, patch_ext.nodes().into_iter());
        // patch_ext_plot.show_html("patch_ext.html");
    }
    
    #[test]
    fn find_patch() {
        // Refine mesh
        let mut msh = MSH.clone();
        msh.lin_subd();
        msh.lin_subd();

        // Find patches
        let face_id = 42;
        let face = msh.faces[face_id];
        let patch = patch::NodeConnectivity::find(&msh, face);

        // Plot nodes of patch
        let nodes_plot = plot::plot_nodes(&msh, patch.as_slice().iter().copied());
        nodes_plot.show_html("out/patch_nodes.html");
    }

    #[test]
    fn patch() {
        // Refine mesh
        let mut msh = MSH.clone();
        msh.lin_subd();
        msh.lin_subd();

        // Find patches
        let mut regular = vec![];
        let mut boundary = vec![];
        let mut corner = vec![];
        let mut irregular = vec![];
        for (face, patch) in msh.patches().enumerate() {
            match patch.nodes {
                NodeConnectivity::Regular(_) => regular.push(face),
                NodeConnectivity::Boundary(_) => boundary.push(face),
                NodeConnectivity::Corner(_) => corner.push(face),
                NodeConnectivity::Irregular(_) => irregular.push(face),
            }
        }

        println!("Regular patches {:?}", regular);
        println!("Boundary patches {:?}", boundary);
        println!("Corner patches {:?}", corner);
        println!("Irregular patches {:?}", irregular);
    }

    #[test]
    fn boundary() {
        // Refine mesh
        let mut msh = MSH.clone();
        msh.lin_subd();
        msh.lin_subd();

        let face_id = 10;
        let face = msh.faces[face_id];
        let patch = msh.find_patch(face);

        let nodes_plot = plot::plot_nodes(&msh, patch.nodes.as_slice().iter().copied());
        nodes_plot.show_html("out/patch_bnd_nodes.html");
    }

    #[test]
    fn surf() {
        // Refine mesh
        let mut msh = MSH.clone();
        msh.lin_subd();
        msh.lin_subd();

        let face_id = 10;
        let face = msh.faces[face_id];
        let patch = msh.find_patch(face);
        println!("{:?}", patch.nodes);

        // Evaluation
        let num_eval = 10;

        let patch_eval_plot = plot::plot_patch(&patch, num_eval);
        patch_eval_plot.show_html("out/patch_eval.html");

        let surf_eval_plot = plot::plot_surf(&msh, num_eval);
        surf_eval_plot.show_html("out/surf_eval.html");
    }

    #[test]
    fn catmull_clark_matrix() {
        let n = 5;

        // Normal subd matrix
        let s = catmull_clark::build_mat::<f64>(4) * 16f64;
        println!("Catmull clark matrix in (F1,...,Fn,E1,...,En,V) ordering: {s}");

        let s = catmull_clark::permute_matrix(&s);
        println!("Catmull clark matrix in (V,E1,F1,...,En,Fn) ordering: {s}");

        // S11, S12, S21 and S22
        println!("S11 = {} S12 = {} S21 = {} S22 = {}", *S11, *S12, *S21, *S22);

        // Extended subd matrices
        let (a, a_bar) = catmull_clark::build_extended_mats::<f64>(n);
        println!("Catmull clark extended matrix {a}");
        println!("Catmull clark bigger extended matrix {a_bar}");

        // EV decomposition
        let svd = a.clone().svd_unordered(true, true);
        let u = svd.u.unwrap();
        let vt = svd.v_t.unwrap();
        let e = svd.singular_values;
        let lambda = Matrix::from_diagonal(&e);
        println!("Error ||uu^t - id||_2 = {:e}", (u.clone() * u.transpose() - SMatrix::<f64, 18, 18>::identity()).norm());
        println!("Error ||vv^t - id||_2 = {:e}", (vt.transpose() * vt.clone() - SMatrix::<f64, 18, 18>::identity()).norm());
        println!("Error ||A - U Λ V^T|| = {:e}", (a.clone() - u.clone() * lambda * vt.clone()).norm());

        // Power calculation
        let nsub = 5;
        let lambda_pow = Matrix::from_diagonal(&e.map(|e| e.powi(nsub)));
        println!("Error ||A^n - U Λ^n V^T|| = {:e}", (a.pow(nsub as u32) - u.clone() * lambda_pow * vt).norm());
    }

    #[test]
    fn sub_patch_transform() {
        let u = 0.26;
        let v = 0.24;
        let sub_patches_plot = plot::plot_sub_patch_hierarchy(u, v);
        sub_patches_plot.show_html("out/sub_patches.html");
    }

    #[test]
    fn patch_knots() {
        // Refine mesh
        let mut msh = MSH.clone();
        msh.lin_subd();
        msh.lin_subd();

        let num_plot = 20;
        let min = 1e-5;
        let knots = lin_space(min..=1.0, 4);
        let knot_lines = knots.map(|u| (u, lin_space(min..=1.0, num_plot)));

        let mut plot = Plot::new();
        let face_indices = [0, 16, 32, 48, 64, 1, 2, 3];
        let patches = face_indices.map(|f| msh.find_patch(msh.faces[f]));

        for patch in patches {
            for (u, vs) in knot_lines.clone() {
                let pos = vs.clone().map(|v| patch.eval(u, v)).collect_vec();
                let xs = pos.iter().map(|p| p.x).collect_vec();
                let ys = pos.iter().map(|p| p.y).collect_vec();
                let u_line = Scatter::new(xs, ys);
                plot.add_trace(u_line);

                let pos = vs.map(|v| patch.eval(v, u)).collect_vec();
                let xs = pos.iter().map(|p| p.x).collect_vec();
                let ys = pos.iter().map(|p| p.y).collect_vec();
                let v_line = Scatter::new(xs, ys);
                plot.add_trace(v_line);
            }
        }

        plot.show_html("out/patch_knots.html");
    }

    #[test]
    fn eval_bspline() {
        let num = 50;

        // Plot B-splines
        let mut bspline_plot = Plot::new();
        for i in 0..4 {
            let plot = plot::plot_fn(|t| { basis::bspline(t)[i] }, num);
            bspline_plot.add_traces(plot.data().iter().cloned().collect());
        }
        bspline_plot.show_html("out/bspline.html".to_string());

        // Plot derivative
        let mut bspline_deriv_plot = Plot::new();
        for i in 0..4 {
            let plot = plot::plot_fn(|t| { basis::bspline_deriv(t)[i] }, num);
            bspline_deriv_plot.add_traces(plot.data().iter().cloned().collect());
        }
        bspline_deriv_plot.show_html("out/bspline_deriv.html".to_string());

        // Plot boundary B-splines
        let mut bspline_bnd_plot = Plot::new();
        for i in 0..3 {
            let plot = plot::plot_fn(|t| { basis::bspline_interpolating(t)[i] }, num);
            bspline_bnd_plot.add_traces(plot.data().iter().cloned().collect());
        }
        bspline_bnd_plot.show_html("out/bspline_bnd.html".to_string());
    }

    #[test]
    fn eval_basis() {
        let num_plot = 50;

        let valence = 5;
        let num_reg = 16;
        let num_irr = 2*valence + 8;

        let u_bnd = true;
        let v_bnd = true;
        let num_bnd = match (u_bnd, v_bnd) {
            (false, false) => 16,
            (true, true) => 6,
            _ => 9
        };

        // Plot derivatives of basis functions
        for i in 0..num_reg {
            let basis_deriv_plot = plot::plot_patch_fn_parametric(|u, v| basis::eval_regular_du(u, v)[i], num_plot);
            basis_deriv_plot.show_html(format!("out/basis_deriv_{i}.html"));
        }

        // Plot all boundary basis functions
        for i in 0..num_bnd {
            let basis_bnd_plot = plot::plot_patch_fn_parametric(|u, v| basis::eval_boundary(u, v, false, false)[i], num_plot);
            // basis_bnd_plot.show_html(format!("out/basis_bnd_{i}.html"));
        }

        // Plot all irregular basis functions
        for i in 0..num_irr {
            let basis_irr_plot = plot::plot_patch_fn_parametric(|u, v| basis::eval_irregular(u, v, valence)[i], num_plot);
            // basis_irr_plot.show_html(format!("out/basis_irr_{i}.html"));
        }
    }

    #[test]
    fn quadrature() {
        // Refine mesh
        let mut msh = MSH.clone();
        msh.lin_subd();
        msh.lin_subd();

        let face_id = 25;
        let face = msh.faces[face_id];
        let patch = msh.find_patch(face);

        // Approximate area by parallelogram
        let nodes = face.map(|n| msh.node(n));
        let ab = nodes[1] - nodes[0];
        let ad = nodes[3] - nodes[0];
        let area_lin = (ab.x * ad.y - ab.y * ad.x).abs();

        // Calculate area by quadrature
        let area_quad = patch.calc_area();

        // Compare values
        println!("Parallelogram area = {area_lin}");
        println!("Quadrature area = {area_quad}");
        println!("Relative error = {:.3}%", (area_quad - area_lin).abs() / area_lin * 100.0);

        // Calculate area of surface
        let area_surf = msh.calc_area();
        println!("Total area of surface = {area_surf:.3}");

        // Precomputation of basis functions
        let num_quad = 2;
        let quad = GaussLegendrePatch::new(num_quad).unwrap();
        let basis_eval = BasisEval::from(&patch, quad.clone());
        let jacobian_eval = JacobianEval::from(&patch, quad.clone());

        println!("Shape of precomputed basis functions: (num_patch, num_quad, num_basis) = ({}, {}, {})",
                 msh.faces.len(),
                 basis_eval.quad_to_basis.len(),
                 basis_eval.quad_to_basis[0].len());

        // Comparison with quad crate
        let int1 = quad.integrate_pullback(basis_eval.quad_to_basis.iter().map(|_| 1.0).collect(), &jacobian_eval);
        let int2 = patch.integrate_pullback(|_, _| 1.0, num_quad);
        println!("Integral using precomputed patch quadrature = {int1}");
        println!("Integral using quadrature crate = {int2}");
        println!("Relative integral error = {:.3}%", (int2 - int1).abs() / int2 * 100.0);
    }

    #[test]
    fn iga_fn() {
        // Refine mesh
        let mut msh = MSH.clone();
        msh.lin_subd();
        msh.lin_subd();
        msh.lin_subd();

        // Define function on patch
        let patch = msh.patches().next().unwrap();
        let f = |p: Point2<f64>| p.x * p.y;
        let fh = IgaFn::from_fn(&msh, f);

        // Calculate L2 error on patch
        let f_eval = |u: f64, v: f64| f(patch.eval(u, v));
        let fh_eval = |u: f64, v: f64| fh.eval_pullback(&patch, u, v);

        let num = 20;
        let uv_range = lin_space(1e-5..=1.0, num);
        let err = uv_range.clone().cartesian_product(uv_range.clone())
            .map(|(u,v)| (f_eval(u,v) - fh_eval(u,v)).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm = uv_range.clone().cartesian_product(uv_range.clone())
            .map(|(u,v)| f_eval(u,v).powi(2))
            .sum::<f64>()
            .sqrt();

        let err_l2 = patch.integrate_pullback(|u, v| (f_eval(u, v) - fh_eval(u, v)).powi(2), num).sqrt();
        let norm_l2 = patch.integrate_pullback(|u, v| f_eval(u, v).powi(2), num).sqrt();

        println!("Relative mesh error |f - fh|_M / |f|_M = {:.3}%", err / norm * 100.0);
        println!("Relative L2 error ||f - fh||_2 / ||f||_2 = {:.3}%", err_l2/ norm_l2 * 100.0);

        // Plot functions
        let f_plot = plot::plot_patch_fn_parametric(f_eval, num);
        let fh_plot = plot::plot_patch_fn_parametric(fh_eval, num);
        // f_plot.show_html("out/iga_f.html");
        // fh_plot.show_html("out/iga_fh.html");
    }
    
    #[test]
    fn iga_mats() {
        // Refine mesh
        let mut msh = MSH.clone();
        msh.lin_subd();
        msh.lin_subd();
        msh.lin_subd();

        let start = Instant::now();
        let quad = GaussLegendrePatch::new(2).unwrap();
        let b_eval = BasisEval::from_mesh(&msh, quad.clone());
        let p_eval = PointEval::from_mesh(&msh, quad.clone());
        let grad_b_eval = GradEval::from_mesh(&msh, quad.clone());
        let j_eval = JacobianEval::from_mesh(&msh, quad.clone());
        // let mat = op_u_v(&msh, &b_eval, &j_eval);
        // let mat = op_gradu_gradv(&msh, &grad_b_eval, &j_eval);
        let mat = op_f_v(&msh, |p| 1.0, &b_eval, &p_eval, &j_eval);
        let time = start.elapsed();

        println!("Building matrix of size {} took {:?}", mat.nrows(), time);
    }

}