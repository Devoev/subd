use std::collections::HashMap;
use crate::cells::node::NodeIdx;
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::traits::MeshTopology;
use nalgebra::{matrix, Matrix5x4, RealField};
use nalgebra_sparse::CooMatrix;
use std::sync::LazyLock;
use itertools::Itertools;
use crate::cells::line_segment::UndirectedEdge;
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
type EdgeMidpoints = HashMap<UndirectedEdge, NodeIdx>;

/// Face to midpoint index map.
type FaceMidpoints = HashMap<QuadNodes, NodeIdx>;

/// Assembles the global subdivision matrix for the given `quad_msh`.
pub fn assemble_mat<T: RealField, const M: usize>(quad_msh: &QuadVertexMesh<T, M>) -> (CooMatrix<f64>, EdgeMidpoints, FaceMidpoints) {
    let edges = quad_msh.edges().collect_vec();
    let num_nodes = quad_msh.num_nodes();
    let mut mat = CooMatrix::new(quad_msh.num_elems() + edges.len(), num_nodes);
    let mut edge_midpoints: EdgeMidpoints = HashMap::new();
    let mut face_midpoints: FaceMidpoints = HashMap::new();

    // for elem in &quad_msh.elems {
    //     let [a, b, c, d] = elem.nodes();
    //     for (local_idx, &NodeIdx(global_idx)) in elem.nodes().iter().enumerate() {
    //
    //     }
    // }

    for (elem_idx, elem) in quad_msh.elems.iter().enumerate() {
        let [NodeIdx(a), NodeIdx(b), NodeIdx(c), NodeIdx(d)] = elem.nodes();
        mat.push(elem_idx, a, 0.25);
        mat.push(elem_idx, b, 0.25);
        mat.push(elem_idx, c, 0.25);
        mat.push(elem_idx, d, 0.25);
        face_midpoints.insert(*elem, NodeIdx(elem_idx + num_nodes));
    }

    let idx_offset = quad_msh.num_elems();
    for (edge_idx, edge) in edges.into_iter().enumerate() {
        let [NodeIdx(a), NodeIdx(b)] = edge.0;
        mat.push(edge_idx + idx_offset, a, 0.5);
        mat.push(edge_idx + idx_offset, b, 0.5);
        edge_midpoints.insert(edge.into(), NodeIdx(edge_idx + idx_offset + num_nodes));
    }

    (mat, edge_midpoints, face_midpoints)
}