use nalgebra::{center, point};
use std::f64::consts::PI;
use std::time::Instant;
use subd::cells::quad::QuadNodes;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::subd::catmull_clark::refine::do_refine;

fn main() {
    let num_refine = 5;

    let mut msh = make_mesh();
    let start = Instant::now();
    for _ in 0..num_refine {
        do_refine(&mut msh);
    }
    let time = start.elapsed();
    // let plt = plot_faces(&msh, msh.elems.clone().into_iter());
    // plt.show();
    println!("Took {:?} for {num_refine} Catmull-Clark subdivision.", time);
}

fn make_mesh() -> QuadVertexMesh<f64, 2> {
    // Angle between segments
    let phi = 2.0*PI / 5.0;

    // Calc coords
    let mut coords = vec![point![0.0, 0.0]];
    for i in 0..5 {
        let phi_i = phi * i as f64;
        let phi_j = phi * (i + 1) as f64;
        let pi = point![phi_i.cos(), phi_i.sin()];
        let pj = point![phi_j.cos(), phi_j.sin()];
        coords.push(pi);
        coords.push(center(&pi, &pj));
    }

    // Faces
    let faces = vec![
        QuadNodes::from_indices(0, 10, 1, 2),
        QuadNodes::from_indices(0, 2, 3, 4),
        QuadNodes::from_indices(0, 4, 5, 6),
        QuadNodes::from_indices(0, 6, 7, 8),
        QuadNodes::from_indices(0, 8, 9, 10),
    ];
    QuadVertexMesh::new(coords, faces)
}