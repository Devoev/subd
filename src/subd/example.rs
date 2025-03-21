use crate::subd::catmull_clark::{S11, S12};
use crate::subd::mesh::{LogicalMesh, QuadMesh};
use crate::subd::plot::{plot_faces, plot_nodes};
use crate::subd::{basis, catmull_clark};
use nalgebra::point;
use std::ops::Deref;

#[test]
fn run_example() {
    println!("Subdivision IGA example");
    
    let coords_quad = vec![
        point![0.2, 0.0], point![0.9, 0.1], point![1.0, 1.0], point![0.0, 1.0]
    ];
    let faces_quad = vec![[0, 1, 2, 3]];
    

    let coords_fichera = vec![
        point![0.0, 0.0], point![0.0, 0.5], point![0.0, 1.0],
        point![0.5, 0.0], point![0.5, 0.5], point![0.5, 1.0],
        point![1.0, 0.0], point![1.0, 0.5]
    ];
    let faces_fichera = vec![
        [0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6]
    ];
    
    let coords_irregular = vec![
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
    ];
    let faces_irregular = vec![
        [0, 1, 2, 3],
        [0, 3, 4, 5],
        [7, 0, 5, 6],
        [8, 9, 0, 7],
        [9, 10, 1, 0]
    ];

    let mut msh = QuadMesh {
        nodes: coords_irregular,
        logical_mesh: LogicalMesh {
            faces: faces_irregular
        }
    };
    
    msh.lin_subd();
    msh.lin_subd();
    // msh.dual();
    // msh.dual();
    // msh.repeated_averaging(3, 2);

    // Plot of mesh
    let msh_plot = plot_faces(&msh, msh.faces.iter().copied());
    // msh_plot.show_html("msh.html");

    let msh_nodes_plot = plot_nodes(&msh, 0..msh.num_nodes());
    // msh_nodes_plot.show_html("msh_nodes.html");

    // Plot extended patch
    // for face_id in 0..=3 {
    //     let patch = msh.find_patch(msh.faces[face_id]);
    //     let patch_plot = plot_faces(&msh, patch.faces.iter().copied());
    //     patch_plot.show_html(format!("plot_{face_id}.html"));
    // }
    
    // let face_id = 3;
    // let patch = msh.find_patch(msh.faces[face_id]);
    // let patch = patch.sort_by_origin(patch.faces[7][3]);
    
    let face_irr_id = 0;
    // let patch_irr = msh.find_patch(msh.faces[face_irr_id]);
    let patch_ext = msh.find_patch_ext(msh.faces[face_irr_id]);

    // Plots
    // let patch_plot = plot_nodes(&msh, patch.nodes_regular().into_iter());
    // patch_plot.show_html("patch.html");

    // let patch_irr_plot = plot_nodes(&msh, patch_irr.nodes_irregular().into_iter());
    // patch_irr_plot.show_html("patch_irr.html");

    let patch_ext_plot = plot_nodes(&msh, patch_ext.nodes().into_iter());
    patch_ext_plot.show_html("patch_ext.html");
    
    // Test Catmull-Clark
    // msh.catmull_clark();
}

#[test]
fn catmull_clark_matrix() {
    let n = 5;

    // Normal subd matrix
    let s = catmull_clark::build_mat::<f64>(4) * 16f64;
    println!("Catmull clark matrix in (F1,...,Fn,E1,...,En,V) ordering: {s}");
    
    let s = catmull_clark::permute_matrix(&s);
    println!("Catmull clark matrix in (V,E1,F1,...,En,Fn) ordering: {s}");

    // S11 and S12
    println!("S11 = {} and S12 = {}", S11.deref(), S12.deref());

    // Extended subd matrices
    let (a, a_bar) = catmull_clark::build_extended_mats::<f64>(n);
    println!("Catmull clark extended matrix {a}");
    println!("Catmull clark bigger extended matrix {a_bar}");

    // EV decomposition
    let (q, t) = a.schur().unpack();
    let lambda = t.diagonal();
    println!("Eigenvectors {q} and eigenvalues {lambda}");
}

#[test]
fn eval_basis() {
    let u = 0.015;
    let v = 0.613;
    let b = basis::eval_regular(u, v);
    let p = basis::permutation_vec(0, 5);
    println!("Basis functions on irregular patch {}", basis::apply_permutation(5, b, p))
}