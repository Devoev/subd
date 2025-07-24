use nalgebra::{matrix, DMatrix};
use subd::cells::quad::QuadNodes;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::plot::plot_faces;

fn main() {
    // Linear quad mesh
    let mut msh = make_mesh();
    msh = msh.lin_subd().unpack();
    let plt = plot_faces(&msh, msh.elems.clone().into_iter());
    plt.show();
    
    // Catmull Clark subdivision
    let s = subd::subd::catmull_clark::matrices::assemble_global_mat(&msh);
    let mut dense = DMatrix::<f64>::zeros(s.nrows(), s.ncols());
    for (i, j, &v) in s.triplet_iter() {
        dense[(i, j)] = v;
    }
    println!("{}", dense);
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