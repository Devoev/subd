use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::traits::MeshTopology;
use itertools::Itertools;
use nalgebra::{DimName, DimNameDiff, DimNameSub, RealField, U1};
use nalgebra_sparse::CooMatrix;
use crate::cells::chain::Chain;
use crate::cells::topo::{Cell, CellBoundary, OrderedCell, OrientedCell, SubCell};
// todo: replace this with a generic implementation. For that add
//  - generic edges, that provide all methods of NodePair (sorted, start, end...)
//  -

/// Assembles the edge-to-node incidence matrix ot the given `msh`.
pub fn edge_to_node_incidence<T: RealField, const M: usize>(msh: &QuadVertexMesh<T, M>) -> CooMatrix<i8> {
    let edges = msh.edges().collect_vec();
    let num_edges = edges.len();
    let num_nodes = msh.num_nodes();
    
    // Build empty incidence matrix
    let mut mat = CooMatrix::new(num_nodes, num_edges);

    // Iteration over edges (columns)
    for (edge_idx, edge) in edges.iter().enumerate() {
        mat.push(edge.start().0, edge_idx, -1);
        mat.push(edge.end().0, edge_idx, 1);
    }

    mat
}

/// Assembles the face-to-edge incidence matrix of the given `msh`.
pub fn face_to_edge_incidence<T: RealField, const M: usize>(msh: &QuadVertexMesh<T, M>) -> CooMatrix<i8> {
    let edges = msh.edges().collect_vec();
    let num_edges = edges.len();
    let num_faces = msh.num_elems();
    // let mut mat = CooMatrix::new(num_edges, num_faces);
    //
    // for (face_idx, face) in msh.elem_iter().enumerate() {
    //     let face_edges = face.edges();
    //     for face_edge in face_edges {
    //         if let Some(edge_idx) = edges.iter().position(|&edge| edge == face_edge) {
    //             mat.push(edge_idx, face_idx, 1);
    //         } else if let Some(edge_idx) = edges.iter().position(|&edge| edge == face_edge.reversed()) {
    //             mat.push(edge_idx, face_idx, -1);
    //         }
    //     }
    // }
    //
    // mat
    incidence(num_faces, num_edges, msh.elems.clone().into_iter(), edges)
}

/// Assembles the `K` to `K-1` incidence matrix.
pub fn incidence<C, K>(num_cells: usize, num_sub_cells: usize, cell_iter: impl Iterator<Item = C>, sub_cells: Vec<SubCell<K, C>>) -> CooMatrix<i8>
    where C: CellBoundary<K>, 
          K: DimName + DimNameSub<U1> + DimNameSub<K>,
          SubCell<K, C>: OrientedCell<DimNameDiff<K, U1>> + Eq, <K as DimNameSub<U1>>::Output: DimNameSub<<K as DimNameSub<U1>>::Output>
{
    // Build empty incidence matrix
    let mut mat = CooMatrix::new(num_sub_cells, num_cells);

    // Iteration over K-cells (columns)
    for (cell_idx, cell) in cell_iter.enumerate() {
        let boundary = cell.boundary();

        // Iteration over K-1-subcells (rows)
        for sub_cell in boundary.cells() {
            // Find index of `sub_cell` is in the mesh, by topological equality
            if let Some((sub_idx, other)) = sub_cells.iter().find_position(|other| other.topo_eq(sub_cell)) {
                // If (global) orientation is positive add +1, else -1
                if other.orientation_eq(sub_cell) {
                    mat.push(sub_idx, cell_idx, 1);
                } else {
                    mat.push(sub_idx, cell_idx, -1);
                }
            }
        }
    }

    mat
}