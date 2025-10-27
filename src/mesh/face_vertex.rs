//! Special cases of an [`ElemVertexMesh`] for `Dim = 2`, i.e. where the elements are faces.

use crate::cells::chain::Chain;
use crate::cells::quad::QuadNodes;
use crate::cells::traits::{Cell, CellBoundary, CellConnectivity, OrderedCell, OrientedCell};
use crate::mesh::elem_vertex::ElemVertexMesh;
use itertools::Itertools;
use nalgebra::{RealField, Scalar, U2};
use std::hash::Hash;
use crate::cells::node::Node;

/// A face-vertex mesh with quadrilateral faces.
pub type QuadVertexMesh<T, const M: usize> = ElemVertexMesh<T, QuadNodes, M>;

impl <T, F, const M: usize> ElemVertexMesh<T, F, M>
    where T: Scalar,
          F: CellBoundary<Dim = U2>,
          F::Node: Eq + Hash,
          F::SubCell: OrderedCell + OrientedCell + CellConnectivity + Clone + Eq + Hash
{
    /// Returns an iterator over all unique and sorted edges in this mesh.
    pub fn edges(&self) -> impl Iterator<Item = F::SubCell> + '_ {
        self.cell_iter()
            .flat_map(|face| face.boundary().cells().to_owned())
            .map(|edge: F::SubCell| edge.sorted())
            .unique()
    }

    /// Returns an iterator over all *open* edges,
    /// i.e. edges that are connected to only *one* face.
    pub fn open_edges(&self) -> impl Iterator<Item = F::SubCell> + '_ {
        self.cell_iter()
            .flat_map(|face| face.boundary().cells().to_owned())
            .map(|edge: F::SubCell| edge.sorted())
            .counts()
            .into_iter()
            .filter_map(|(edge, num)| (num == 1).then_some(edge))
    }

    /// Returns all edges connected to the given `node`.
    pub fn edges_of_node(&self, node: F::Node) -> impl Iterator<Item = F::SubCell> + '_ {
        self.edges().filter(move |edge| edge.contains_node(node))
    }

    /// Returns all faces connected to the given (undirected) `edge`.
    pub fn faces_of_edge(&self, edge: F::SubCell) -> impl Iterator<Item = &F> + '_ {
        self.cell_iter()
            .filter(move |face| {
                let edges = face.boundary().cells().to_owned();
                edges.contains(&edge) || edges.contains(&edge.reversed())
            })
    }

    /// Calculates the valence of the given `node`, i.e. the number of edges connected to the node.
    pub fn valence(&self, node: F::Node) -> usize {
        self.edges_of_node(node).count()
    }

    // todo: this method is probably inefficient, because it iterates over ALL open edges
    /// Returns `true` if the given `node` is a boundary node,
    /// i.e. it is part of an open edge.
    pub fn is_boundary_node(&self, node: F::Node) -> bool {
        self.open_edges().any(|edge| edge.contains_node(node))
    }

    /// Returns an iterator over all boundary nodes.
    pub fn boundary_nodes(&self) -> impl Iterator<Item = F::Node> + '_ {
        self.open_edges()
            .flat_map(|edge| edge.nodes().to_vec())
            .unique()
    }
}

impl<T: Scalar, const M: usize> QuadVertexMesh<T, M> {
    /// Returns `true` if the `node` is regular.
    pub fn is_regular_node(&self, node: Node) -> bool {
        self.valence(node) == 4
    }
    
    /// Returns `true` if the face is regular.
    pub fn is_regular_face(&self, face: QuadNodes) -> bool {
        face.nodes().iter().all(|node| self.valence(*node) == 4)
    }

    /// Finds the irregular node of the given `face`, if any exists.
    pub fn irregular_node_of_face(&self, face: QuadNodes) -> Option<Node> {
        face.nodes()
            .into_iter()
            .find(|&v| self.valence(v) != 4)
    }

    /// Finds all boundary nodes of the given `face`,
    /// i.e. all irregular nodes, assuming the face is a boundary face.
    pub fn boundary_nodes_of_face(&self, face: QuadNodes) -> Vec<Node> {
        face.nodes().into_iter().filter(|&v| self.valence(v) != 4).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::edge::{DirectedEdge, UndirectedEdge};
    use nalgebra::SMatrix;
    use std::collections::HashSet;


    /// Constructs the (irregular) quad mesh
    /// ```text
    ///   2 --- 3 --- 4
    ///   |  0  |  1  |
    ///   1 --- 0 --- 5
    ///  ╱    ╱ |  2  |
    /// 10 4 ╱3 7 --- 6
    ///  ╲  ╱  ╱
    ///   9 - 8
    /// ```
    /// of valence `n=5` with all-zero control points.
    fn setup() -> QuadVertexMesh<f64, 2> {
        let faces = vec![
            QuadNodes::new(1, 0, 3, 2),
            QuadNodes::new(0, 5, 4, 3),
            QuadNodes::new(7, 6, 5, 0),
            QuadNodes::new(9, 8, 7, 0),
            QuadNodes::new(9, 0, 1, 10),
        ];

        QuadVertexMesh::from_coords_matrix(SMatrix::<f64, 11, 2>::zeros(), faces)
    }

    #[test]
    fn faces_of_edge() {
        let msh = setup();

        // Edge 0 -> 5 and 5 -> 0
        let faces_exp = HashSet::from([&msh.cells[1], &msh.cells[2]]);
        let edge = DirectedEdge([0, 5]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert_eq!(faces, faces_exp);
        
        let edge = DirectedEdge([5, 0]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert_eq!(faces, faces_exp);
        
        // Edge 0 -> 9 and 9 -> 0
        let faces_exp = HashSet::from([&msh.cells[3], &msh.cells[4]]);
        let edge = DirectedEdge([0, 9]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert_eq!(faces, faces_exp);

        let edge = DirectedEdge([9, 0]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert_eq!(faces, faces_exp);
        
        // Edge 7 -> 6 and 6 -> 7
        let faces_exp = HashSet::from([&msh.cells[2]]);
        let edge = DirectedEdge([7, 6]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert_eq!(faces, faces_exp);

        let edge = DirectedEdge([6, 7]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert_eq!(faces, faces_exp);
        
        // Edge 1 -> 2 and 2 -> 1
        let faces_exp = HashSet::from([&msh.cells[0]]);
        let edge = DirectedEdge([1, 2]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert_eq!(faces, faces_exp);

        let edge = DirectedEdge([2, 1]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert_eq!(faces, faces_exp);
        
        // Edge 4 -> 0 (non existent)
        let edge = DirectedEdge([4, 0]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert!(faces.is_empty());
    }

    #[test]
    fn boundary() {
        let msh = setup();

        // Boundary edges
        let edges: HashSet<UndirectedEdge> = msh.open_edges().map(|edge| edge.into()).collect();
        let edges_exp = HashSet::from([
            UndirectedEdge::new(1, 2),
            UndirectedEdge::new(2, 3),
            UndirectedEdge::new(3, 4),
            UndirectedEdge::new(4, 5),
            UndirectedEdge::new(5, 6),
            UndirectedEdge::new(6, 7),
            UndirectedEdge::new(7, 8),
            UndirectedEdge::new(8, 9),
            UndirectedEdge::new(9, 10),
            UndirectedEdge::new(10, 1),
        ]);
        assert_eq!(edges, edges_exp);

        // Boundary nodes
        let nodes: HashSet<Node> = msh.boundary_nodes().collect();
        let nodes_exp = HashSet::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(nodes, nodes_exp);
    }
}