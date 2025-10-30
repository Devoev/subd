use std::io;
use nalgebra::point;
use subd::cells::quad::QuadNodes;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::plot::plot_faces;
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
    let msh_regular = QuadVertexMesh::new(coords.clone(), quads);

    plot_faces(&msh_regular, msh_regular.cell_iter().copied()).show();

    // fixme: this panics, because valence 3 case isn't implemented yet
    let msh_regular = CatmarkMesh::from(msh_regular);

    Ok(())
}