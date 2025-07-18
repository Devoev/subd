use nalgebra::matrix;
use std::hint::black_box;
use std::time::Instant;
use subd::cells::quad::QuadNodes;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::plot::plot_faces;

fn main() {
    let num_refine = 11;

    let mut msh = make_mesh();
    let start = Instant::now();
    for _ in 0..num_refine {
        msh = black_box(msh.lin_subd().unpack());
    }
    let time = start.elapsed();
    // let plt = plot_faces(&msh, msh.elems.clone().into_iter());
    // plt.show();

    let mut msh = make_mesh();
    let start = Instant::now();
    for _ in 0..num_refine {
        // todo!("This test is disabled currently")
        // black_box(msh.refine_alt());
    }
    let time_alt = start.elapsed();
    // let plt = plot_faces(&msh.0, msh.0.elems.clone().into_iter());
    // plt.show();

    println!(
        "Took {:?} for {num_refine} linear mesh refinements.",
        time
    );
    println!(
        "Took {:?} for {num_refine} linear mesh refinements (alt method).",
        time_alt
    );
    println!(
        "alt is {} % slower than original method",
        (time_alt.as_secs_f64() - time.as_secs_f64()) / time.as_secs_f64()
            * 100.0
    )
}

fn make_mesh() -> QuadVertexMesh<f64, 2> {
    // Define mesh
    let coords_square = matrix![
            0.0, 0.0, 1.0, 1.0;
            0.0, 1.0, 1.0, 0.0
        ].transpose();
    let quads = vec![QuadNodes::from_indices(0, 1, 2, 3)];
    QuadVertexMesh::from_matrix(coords_square, quads)
}