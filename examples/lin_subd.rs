use nalgebra::matrix;
use std::hint::black_box;
use std::time::Instant;
use subd::cells::quad::QuadTopo;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::plot::plot_faces;
use subd::subd::lin_subd::LinSubd;

fn main() {
    let num_refine = 11;

    let mut msh = make_mesh();
    let start = Instant::now();
    for _ in 0..num_refine {
        black_box(msh.refine());
    }
    let time = start.elapsed();
    // let plt = plot_faces(&msh.0, msh.0.elems.clone().into_iter());
    // plt.show();

    let mut msh = make_mesh();
    let start = Instant::now();
    for _ in 0..num_refine {
        todo!("This test is disabled currently")
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

fn make_mesh() -> LinSubd<f64, 2> {
    // Define mesh
    let coords_square = matrix![
            0.0, 0.0, 1.0, 1.0;
            0.0, 1.0, 1.0, 0.0
        ].transpose();
    let quads = vec![QuadTopo::from_indices(0, 1, 2, 3)];
    let quad_msh = QuadVertexMesh::from_matrix(coords_square, quads);
    LinSubd(quad_msh)
}