use std::collections::HashMap;
use crate::cells::node::Node;
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::traits::MeshTopology;
use nalgebra::{matrix, Matrix5x4, RealField};
use nalgebra_sparse::CooMatrix;
use std::sync::LazyLock;
use itertools::Itertools;
use crate::cells::edge::UndirectedEdge;
use crate::cells::quad::QuadNodes;

// todo: maybe change the code below to use the local subdivision matrix S

/// The `5✕4` local subdivision matrix.
static S: LazyLock<Matrix5x4<f64>> = LazyLock::new(|| {
   matrix![
       0.5, 0.5, 0.0, 0.0;
       0.0, 0.5, 0.5, 0.0;
       0.0, 0.0, 0.5, 0.5;
       0.5, 0.0, 0.0, 0.5;
       0.25, 0.25, 0.25, 0.25
   ]
});

// todo: consider different data structures for midpoints
//  maybe index-vectors to directly construct incidence matrices?

/// Edge to midpoint index map.
type EdgeMidpoints = HashMap<UndirectedEdge, Node>;

/// Face to midpoint index map.
type FaceMidpoints = HashMap<QuadNodes, Node>;

/// Assembles the global subdivision matrix for the given `quad_msh`.
pub fn assemble_global_mat<T: RealField, const M: usize>(quad_msh: &QuadVertexMesh<T, M>) -> (CooMatrix<f64>, EdgeMidpoints, FaceMidpoints) {
    let edges = quad_msh.edges().collect_vec();
    let num_nodes = quad_msh.num_nodes();
    let mut mat = CooMatrix::new(quad_msh.num_elems() + edges.len() + num_nodes, num_nodes);
    let mut edge_midpoints: EdgeMidpoints = HashMap::new();
    let mut face_midpoints: FaceMidpoints = HashMap::new();

    // Apply no node smoothing = identity mapping
    for node_idx in 0..num_nodes {
        mat.push(node_idx, node_idx, 1.0)
    }

    // Apply face-midpoint stencil
    let mut idx_offset = num_nodes;
    for (face_idx, face) in (&quad_msh.cells).into_elem_iter().enumerate() {
        let [a, b, c, d] = face.nodes();
        mat.push(face_idx + idx_offset, a, 0.25);
        mat.push(face_idx + idx_offset, b, 0.25);
        mat.push(face_idx + idx_offset, c, 0.25);
        mat.push(face_idx + idx_offset, d, 0.25);
        face_midpoints.insert(*face, face_idx + idx_offset);
    }

    // Apply edge-midpoint stencil
    idx_offset += quad_msh.num_elems();
    for (edge_idx, edge) in edges.into_iter().enumerate() {
        let [a, b] = edge.0;
        mat.push(edge_idx + idx_offset, a, 0.5);
        mat.push(edge_idx + idx_offset, b, 0.5);
        edge_midpoints.insert(edge.into(), edge_idx + idx_offset);
    }

    (mat, edge_midpoints, face_midpoints)
}