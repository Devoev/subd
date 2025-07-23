use crate::cells::node::NodeIdx;
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::traits::MeshTopology;
use nalgebra::{matrix, Matrix5x4, RealField};
use nalgebra_sparse::CooMatrix;
use std::sync::LazyLock;
use itertools::Itertools;

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

/// Assembles the global subdivision matrix for the given `quad_msh`.
pub fn assemble_mat<T: RealField, const M: usize>(quad_msh: &QuadVertexMesh<T, M>) -> CooMatrix<f64> {
    let edges = quad_msh.edges().collect_vec();
    let mut mat = CooMatrix::new(quad_msh.num_elems() + edges.len(), quad_msh.num_nodes());

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
    }

    let idx_offset = quad_msh.num_elems();
    for (edge_idx, edge) in edges.into_iter().enumerate() {
        let [NodeIdx(a), NodeIdx(b)] = edge.0;
        mat.push(edge_idx + idx_offset, a, 0.5);
        mat.push(edge_idx + idx_offset, b, 0.5);
    }

    mat
}