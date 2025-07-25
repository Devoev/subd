use crate::mesh::face_vertex::QuadVertexMesh;
use crate::subd::catmull_clark::matrices::assemble_global_mat;
use nalgebra::Point;
use nalgebra_sparse::CsrMatrix;
use crate::cells::node::NodeIdx;
use crate::cells::quad::QuadNodes;

/// Refines the given `quad_msh` using the global subdivision matrix.
pub fn do_refine<const M: usize>(quad_msh: &mut QuadVertexMesh<f64, M>) {
    // Refine coords
    let (s, face_midpoints, edge_midpoints) = assemble_global_mat(quad_msh);
    let s = CsrMatrix::from(&s);
    let c = quad_msh.coords_matrix();
    let c_subd = s * &c;

    // Update coords
    quad_msh.coords.clear();
    for point_coords in c_subd.row_iter() {
        quad_msh.coords.push(Point::from(point_coords.transpose()));
    }
    
    // Update connectivity
    let mut refined_faces = Vec::<QuadNodes>::new();
    let mut add_face_nodes = |a: NodeIdx, b: NodeIdx, c: NodeIdx, d: NodeIdx| {
        refined_faces.push(QuadNodes([a, b, c, d]))
    };
    for elem in &quad_msh.elems {
        // Get corner nodes
        let [a, b, c, d] = elem.nodes();

        // Get edge midpoints
        let [ab, bc, cd, da] = elem.undirected_edges()
            .map(|edge| edge_midpoints[&edge]);

        // Get face midpoint
        let m = face_midpoints[elem];

        // Add refined faces
        add_face_nodes(a, ab, m, da);
        add_face_nodes(ab, b, bc, m);
        add_face_nodes(m, bc, c, cd);
        add_face_nodes(da, m, cd, d);
    }

    // Update faces
    quad_msh.elems = refined_faces
}