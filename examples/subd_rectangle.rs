use iter_num_tools::lin_space;
use nalgebra::point;
use std::io;
use subd::cells::quad::QuadNodes;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::plot::{plot_faces, plot_fn_msh};
use subd::subd::catmull_clark::mesh::CatmarkMesh;

fn main() -> io::Result<()> {
    // Define regular mesh
    let mut coords = vec![
        point![0.0, 0.0],
        point![1.0, 0.0],
        point![1.0, 1.0],
        point![0.0, 1.0]
    ];

    let quads = vec![QuadNodes::new(0, 1, 2, 3)];
    let mut msh_regular = QuadVertexMesh::new(coords.clone(), quads);
    msh_regular = msh_regular.lin_subd().unpack();

    plot_faces(&msh_regular, msh_regular.cell_iter().copied()).show();
    plot_fn_msh(&msh_regular, &|_cell, _uv| 1.0, 10, |_elem, num| {
        (lin_space(0.0..=1.0, num).collect(), lin_space(0.0..=1.0, num).collect())
    }).show();

    // Define irregular mesh (valence 3)
    coords.extend_from_slice(&[
        point![0.3, 0.3],
        point![0.7, 0.3],
        point![0.7, 0.7],
        point![0.3, 0.7],
    ]);

    let quads = vec![
        QuadNodes::new(0, 1, 5, 4),
        QuadNodes::new(5, 1, 2, 6),
        QuadNodes::new(7, 6, 2, 3),
        QuadNodes::new(0, 4, 7, 3),
        QuadNodes::new(4, 5, 6, 7),
    ];
    let msh_irregular = QuadVertexMesh::new(coords.clone(), quads).lin_subd().unpack();

    // plot_faces(&msh_irregular, msh_irregular.cell_iter().copied()).show();

    // fixme: this panics, because irregular boundaries are not implemented yet
    let msh_irregular = CatmarkMesh::from(msh_irregular);

    // fixme: this panics, because valence = 3 has not been implemented yet for basis evaluation
    plot_fn_msh(&msh_irregular, &|_cell, _uv| 1.0, 10, |_elem, num| {
        (lin_space(0.0..=1.0, num).collect(), lin_space(0.0..=1.0, num).collect())
    }).show();

    Ok(())
}