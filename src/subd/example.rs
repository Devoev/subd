use crate::subd::catmull_clark::{S11, S12, S21, S22};
use crate::subd::mesh::{Face, LogicalMesh, QuadMesh};
use crate::subd::plot::{plot_faces, plot_nodes, plot_patch, plot_sub_patch_hierarchy, plot_surf};
use crate::subd::{basis, catmull_clark, plot};
use nalgebra::{point, Matrix, Point2, SMatrix};
use std::sync::LazyLock;
use itertools::Itertools;
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
static FACES_STASR: LazyLock<Faces> = LazyLock::new(|| {
    vec![
        [0, 1, 2, 3],
        [0, 3, 4, 5],
        [7, 0, 5, 6],
        [8, 9, 0, 7],
        [9, 10, 1, 0]
    ]
});

/// The used quad mesh.
static MSH: LazyLock<QuadMesh<f64>> = LazyLock::new(|| {
    QuadMesh {
        nodes: COORDS_STAR.clone(),
        logical_mesh: LogicalMesh {
            faces: FACES_STASR.clone()
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
    let same_faces = patch2.faces.iter().sorted().collect_vec() == patch1.faces.iter().sorted().collect_vec();
    let same_center = patch1.center == patch2.center;
    let same_nodes = patch1.nodes_regular().iter().sorted().collect_vec() == patch2.nodes_regular().iter().sorted().collect_vec();
    if !same_faces || !same_center || !same_nodes {
        eprintln!("Faces, center face or nodes are not the same! \
            faces = {same_faces}, center = {same_center}, nodes = {same_nodes}");
    }

    // Plot nodes of patch
    let nodes_plot = plot_nodes(&msh, patch2.nodes_regular().into_iter());
    nodes_plot.show_html("out/patch_nodes.html");
}

#[test]
fn surf() {
    // Refine mesh
    let mut msh = MSH.clone();
    msh.lin_subd();
    msh.lin_subd();

    let face_id = 2;
    let face = msh.faces[face_id];
    let patch = Patch::find(&msh, face, face[0]);

    // Evaluation
    let num_eval = 10;

    let patch_eval_plot = plot_patch(patch, num_eval);
    patch_eval_plot.show_html("patch_eval.html");

    let surf_eval_plot = plot_surf(&msh, num_eval);
    surf_eval_plot.show_html("surf_eval.html");
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
fn eval_basis() {
    // let u = 0.215;
    // let v = 0.613;
    // let b_reg = basis::eval_regular(u, v);
    // let b = basis::eval_irregular(u, v);
    // println!("Basis on regular patch {b_reg}");
    // println!("Basis on irregular patch {b}");

    let num = 50;
    let valence = 5;
    let b_idx = 0;
    // let basis_reg_plot = plot::plot_fn(|u, v| basis::eval_regular(u, v)[b_idx], num);
    // basis_reg_plot.show_html("basis.html");

    // Plot all irregular basis functions
    for i in 0..18 {
        let basis_irr_plot = plot::plot_fn(|u, v| basis::eval_irregular(u, v, valence)[i], num);
        basis_irr_plot.show_html(format!("basis_irr_{i}.html"));
    }
}