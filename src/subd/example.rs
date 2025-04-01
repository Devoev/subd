use std::f64::consts::PI;
use crate::subd::catmull_clark::{S11, S12, S21, S22};
use crate::subd::mesh::{Face, LogicalMesh, QuadMesh};
use crate::subd::plot::{plot_faces, plot_fn, plot_nodes, plot_patch, plot_sub_patch_hierarchy, plot_surf};
use crate::subd::{basis, catmull_clark, plot};
use nalgebra::{center, point, Matrix, Point2, SMatrix};
use std::sync::LazyLock;
use itertools::Itertools;
use plotly::Plot;
use crate::subd::patch::Patch;

/// Vector of coordinates in 2D.
type Coords = Vec<Point2<f64>>;

/// Vector of faces.
type Faces = Vec<Face>;

/// Rectangle coordinates.
static COORDS_QUAD: LazyLock<Coords> = LazyLock::new(|| {
    vec![point![0.2, 0.0], point![0.9, 0.1], point![1.0, 1.0], point![0.0, 1.0]]
});

/// Rectangle faces.
static FACES_QUAD: LazyLock<Faces> = LazyLock::new(|| {
    vec![[0, 1, 2, 3]]
});

/// Fichera coordinates.
static COORDS_FICHERA: LazyLock<Coords> = LazyLock::new(|| {
    vec![
        point![0.0, 0.0], point![0.0, 0.5], point![0.0, 1.0],
        point![0.5, 0.0], point![0.5, 0.5], point![0.5, 1.0],
        point![1.0, 0.0], point![1.0, 0.5]
    ]
});

/// Fichera faces.
static FACES_FICHERA: LazyLock<Faces> = LazyLock::new(|| {
    vec![[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6]]
});

/// Star coordinates.
static COORDS_STAR: LazyLock<Coords> = LazyLock::new(|| {
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
static FACES_STAR: LazyLock<Faces> = LazyLock::new(|| {
    vec![
        [0, 1, 2, 3],
        [0, 3, 4, 5],
        [7, 0, 5, 6],
        [8, 9, 0, 7],
        [9, 10, 1, 0]
    ]
});

/// Pentagon coordinates.
static COORDS_PENTAGON: LazyLock<Coords> = LazyLock::new(|| {
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
static FACES_PENTAGON: LazyLock<Faces> = LazyLock::new(|| {
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
    let msh_plot = plot_faces(&msh, msh.faces.iter().copied());
    msh_plot.show_html("out/msh.html");

    let msh_nodes_plot = plot_nodes(&msh, 0..msh.num_nodes());
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
fn patch() {
    // Refine mesh
    let mut msh = MSH.clone();
    msh.lin_subd();
    msh.lin_subd();

    // Find patches
    let face_id = 2;
    let face = msh.faces[face_id];
    let patch1 = Patch::find(&msh, face, face[0]);
    let patch2 = Patch::find(&msh, face, face[1]);

    // Test if patches are the same
    let same_faces = patch2.faces().iter().sorted().collect_vec() == patch1.faces().iter().sorted().collect_vec();
    let same_center = patch1.center() == patch2.center();
    let same_nodes = patch1.nodes().iter().sorted().collect_vec() == patch2.nodes().iter().sorted().collect_vec();
    if !same_faces || !same_center || !same_nodes {
        eprintln!("Faces, center face or nodes are not the same! \
            faces = {same_faces}, center = {same_center}, nodes = {same_nodes}");
    }

    // Plot nodes of patch
    let nodes_plot = plot_nodes(&msh, patch2.nodes().into_iter());
    nodes_plot.show_html("out/patch_nodes.html");
}

#[test]
fn boundary() {
    // Refine mesh
    let mut msh = MSH.clone();
    msh.lin_subd();
    msh.lin_subd();

    let face_id = 48;
    let face = msh.faces[face_id];
    let patch = msh.find_patch(face);

    let nodes_plot = plot_nodes(&msh, patch.nodes().iter().copied());
    nodes_plot.show_html("out/patch_bnd_nodes.html");
}

#[test]
fn surf() {
    // Refine mesh
    let mut msh = MSH.clone();
    msh.lin_subd();
    msh.lin_subd();

    let face_id = 48;
    let face = msh.faces[face_id];
    let patch = msh.find_patch(face);

    // Evaluation
    let num_eval = 10;

    let patch_eval_plot = plot_patch(patch, num_eval);
    // patch_eval_plot.show_html("out/patch_eval.html");

    let surf_eval_plot = plot_surf(&msh, num_eval);
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
    let sub_patches_plot = plot_sub_patch_hierarchy(u, v);
    sub_patches_plot.show_html("out/sub_patches.html");
}

#[test]
fn eval_bspline() {
    let num = 50;

    // Plot B-splines
    let mut bspline_plot = Plot::new();
    for i in 0..4 {
        let plot = plot_fn(|t| { basis::bspline(t)[i] }, num);
        bspline_plot.add_traces(plot.data().iter().cloned().collect());
    }
    bspline_plot.show_html("out/bspline.html".to_string());

    // Plot derivative
    let mut bspline_deriv_plot = Plot::new();
    for i in 0..4 {
        let plot = plot_fn(|t| { basis::bspline_deriv(t)[i] }, num);
        bspline_deriv_plot.add_traces(plot.data().iter().cloned().collect());
    }
    bspline_deriv_plot.show_html("out/bspline_deriv.html".to_string());

    // Plot boundary B-splines
    let mut bspline_bnd_plot = Plot::new();
    for i in 0..3 {
        let plot = plot_fn(|t| { basis::bspline_interpolating(t)[i] }, num);
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
        let basis_deriv_plot = plot::plot_surf_fn(|u, v| basis::eval_regular_du(u, v)[i], num_plot);
        basis_deriv_plot.show_html(format!("out/basis_deriv_{i}.html"));
    }

    // Plot all boundary basis functions
    for i in 0..num_bnd {
        let basis_bnd_plot = plot::plot_surf_fn(|u, v| basis::eval_boundary(u, v, false, false)[i], num_plot);
        // basis_bnd_plot.show_html(format!("out/basis_bnd_{i}.html"));
    }

    // Plot all irregular basis functions
    for i in 0..num_irr {
        let basis_irr_plot = plot::plot_surf_fn(|u, v| basis::eval_irregular(u, v, valence)[i], num_plot);
        // basis_irr_plot.show_html(format!("out/basis_irr_{i}.html"));
    }
}

#[test]
fn quadrature() {
    // Refine mesh
    let mut msh = MSH.clone();
    msh.lin_subd();
    msh.lin_subd();

    let face_id = 3;
    let face = msh.faces[face_id];
    let patch = msh.find_patch(face);
    
    let j = patch.eval_jacobian(0.0, 0.0);
    println!("{}", j.determinant());
}