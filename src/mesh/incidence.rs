use crate::cells::chain::Chain;
use crate::cells::topo::{Cell, CellBoundary, OrientedCell, SubCell};
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::traits::MeshTopology;
use itertools::Itertools;
use nalgebra::{DimName, DimNameDiff, DimNameSub, RealField, U1};
use nalgebra_sparse::CooMatrix;
use crate::cells::line_segment::DirectedEdge;
use crate::cells::node::NodeIdx;
// todo: replace this with a generic implementation over the mesh type. For that add
//  - generic edges, that provide all methods of NodePair (sorted, start, end...)
//  -

/// Assembles the edge-to-node incidence matrix ot the given `msh`.
pub fn edge_to_node_incidence<T: RealField, const M: usize>(msh: &QuadVertexMesh<T, M>) -> CooMatrix<i8> {
    let edges = msh.edges().collect_vec();
    let num_edges = edges.len();
    let num_nodes = msh.num_nodes();

    assemble_incidence_mat(num_edges, num_nodes, edges.into_iter(), |mat, cell, cell_idx| {
        let  DirectedEdge([NodeIdx(start_idx), NodeIdx(end_idx)]) = cell;
        mat.push(start_idx, cell_idx, -1);
        mat.push(end_idx, cell_idx, 1);
    })
}

/// Assembles the face-to-edge incidence matrix of the given `msh`.
pub fn face_to_edge_incidence<T: RealField, const M: usize>(msh: &QuadVertexMesh<T, M>) -> CooMatrix<i8> {
    let edges = msh.edges().collect_vec();
    let num_edges = edges.len();
    let num_faces = msh.num_elems();

    assemble_incidence_mat(num_faces, num_edges, msh.elems.clone().into_iter(), |mat, cell, cell_idx| {
        populate_by_orientation(mat, cell, cell_idx, &edges)
    })
}

/// Assembles the `K` to `K-1` incidence matrix of size `num_sub_cells ✕ num_cells`.
///
/// For each `K`-cell, the incidences in the `i`-th row get populated using `populate_incidence`.
fn assemble_incidence_mat<C, K, F>(num_cells: usize, num_sub_cells: usize, cell_iter: impl Iterator<Item = C>, populate_incidence: F) -> CooMatrix<i8>
    where C: CellBoundary<K>,
          K: DimName + DimNameSub<U1>,
          F: Fn(&mut CooMatrix<i8>, C, usize),
{
    // Build empty incidence matrix
    let mut mat = CooMatrix::new(num_sub_cells, num_cells);

    // Iteration over K-cells (columns)
    for (cell_idx, cell) in cell_iter.enumerate() {
        // Populate column of mat with incidences of `cell`
        populate_incidence(&mut mat, cell, cell_idx);
    }

    mat
}

/// Populates the `cell_idx`-th column with incidences of `cell`,
/// by comparing the orientation of the boundary cells with the global orientation in `sub_cells` .
fn populate_by_orientation<C, K>(mat: &mut CooMatrix<i8>, cell: C, cell_idx: usize, sub_cells: &[SubCell<K, C>])
    where C: CellBoundary<K>,
          K: DimName + DimNameSub<U1> + DimNameSub<K>,
          SubCell<K, C>: OrientedCell<DimNameDiff<K, U1>> + Eq, <K as DimNameSub<U1>>::Output: DimNameSub<<K as DimNameSub<U1>>::Output>
{
    // Get sub-cells in boundary of current `cell`
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