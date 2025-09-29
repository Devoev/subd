//! Special cases of an [`ElemVertexMesh`] for `K = 2`, i.e. where the elements are faces.

use crate::cells::chain::Chain;
use crate::cells::line_segment::LineSegment;
use crate::cells::node::NodeIdx;
use crate::cells::quad::{Quad, QuadNodes};
use crate::cells::topo::{CellToNodes, CellBoundary, Edge2, OrderedCell, OrientedCell};
use crate::mesh::elem_vertex::ElemVertexMesh;
use itertools::Itertools;
use nalgebra::{Point, RealField, U1, U2};
use std::hash::Hash;
use crate::mesh::traits::Mesh;

/// A face-vertex mesh with `2`-dimensional faces [`C`].
pub type FaceVertexMesh<T, C, const M: usize> = ElemVertexMesh<T, C, 2, M>;

/// A face-vertex mesh with quadrilateral faces.
pub type QuadVertexMesh<T, const M: usize> = FaceVertexMesh<T, QuadNodes, M>;

impl <T: RealField, F: CellBoundary<Dim = U2>, const M: usize> FaceVertexMesh<T, F, M>
where Edge2<F>: OrderedCell + OrientedCell + Clone + Eq + Hash
{
    /// Returns an iterator over all unique and sorted edges in this mesh.
    pub fn edges(&self) -> impl Iterator<Item = Edge2<F>> + '_ {
        self.elems.iter()
            .flat_map(|face| face.boundary().cells().to_owned())
            .map(|edge: Edge2<F> | edge.sorted())
            .unique()
    }

    /// Returns an iterator over all *open* edges,
    /// i.e. edges that are connected to only *one* face.
    pub fn open_edges(&self) -> impl Iterator<Item = Edge2<F>> + '_ {
        self.elems.iter()
            .flat_map(|face| face.boundary().cells().to_owned())
            .map(|edge: Edge2<F> | edge.sorted())
            .counts()
            .into_iter()
            .filter_map(|(edge, num)| (num == 1).then_some(edge))
    }

    /// Returns all edges connected to the given `node`.
    pub fn edges_of_node(&self, node: NodeIdx) -> impl Iterator<Item = Edge2<F>> + '_ {
        self.edges().filter(move |edge| edge.contains_node(node))
    }

    /// Returns all faces connected to the given (undirected) `edge`.
    pub fn faces_of_edge(&self, edge: Edge2<F>) -> impl Iterator<Item = &F> + '_ {
        self.elems.iter()
            .filter(move |face| {
                let edges = face.boundary().cells().to_owned();
                edges.contains(&edge) || edges.contains(&edge.reversed())
            })
    }

    /// Calculates the valence of the given `node`, i.e. the number of edges connected to the node.
    pub fn valence(&self, node: NodeIdx) -> usize {
        self.edges_of_node(node).count()
    }

    // todo: this method is probably inefficient, because it iterates over ALL open edges
    /// Returns `true` if the given `node` is a boundary node,
    /// i.e. it is part of an open edge.
    pub fn is_boundary_node(&self, node: NodeIdx) -> bool {
        self.open_edges().any(|edge| edge.contains_node(node))
    }
}

impl<T: RealField, const M: usize> QuadVertexMesh<T, M> {
    /// Returns `true` if the `node` is regular.
    pub fn is_regular_node(&self, node: NodeIdx) -> bool {
        self.valence(node) == 4
    }
    
    /// Returns `true` if the face is regular.
    pub fn is_regular_face(&self, face: QuadNodes) -> bool {
        face.nodes().iter().all(|node| self.valence(*node) == 4)
    }

    /// Finds the irregular node of the given `face`, if any exists.
    pub fn irregular_node_of_face(&self, face: QuadNodes) -> Option<NodeIdx> {
        face.nodes()
            .into_iter()
            .find(|&v| self.valence(v) != 4)
    }

    /// Finds all boundary nodes of the given `face`,
    /// i.e. all irregular nodes, assuming the face is a boundary face.
    pub fn boundary_nodes_of_face(&self, face: QuadNodes) -> Vec<NodeIdx> {
        face.nodes().into_iter().filter(|&v| self.valence(v) != 4).collect()
    }

    /// Returns an iterator over all unique and sorted edges in this mesh.
    pub fn geo_edges(&self) -> impl Iterator<Item=LineSegment<T, M>> + '_ {
        self.edges().map(|edge_top| LineSegment::from_msh(edge_top, self))
    }

    /// Returns an iterator over all faces in this mesh.
    pub fn geo_faces(&self) -> impl Iterator<Item=Quad<T, M>> + '_ {
        self.elems.iter().map(|&face| Quad::from_msh(face, self))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use nalgebra::SMatrix;
    use crate::cells::line_segment::DirectedEdge;
    use super::*;


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
            QuadNodes::from_indices(1, 0, 3, 2),
            QuadNodes::from_indices(0, 5, 4, 3),
            QuadNodes::from_indices(7, 6, 5, 0),
            QuadNodes::from_indices(9, 8, 7, 0),
            QuadNodes::from_indices(9, 0, 1, 10),
        ];

        QuadVertexMesh::from_matrix(SMatrix::<f64, 11, 2>::zeros(), faces)
    }

    #[test]
    fn faces_of_edge() {
        let msh = setup();

        // Edge 0 -> 5 and 5 -> 0
        let faces_exp = HashSet::from([&msh.elems[1], &msh.elems[2]]);
        let edge = DirectedEdge([NodeIdx(0), NodeIdx(5)]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert_eq!(faces, faces_exp);
        
        let edge = DirectedEdge([NodeIdx(5), NodeIdx(0)]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert_eq!(faces, faces_exp);
        
        // Edge 0 -> 9 and 9 -> 0
        let faces_exp = HashSet::from([&msh.elems[3], &msh.elems[4]]);
        let edge = DirectedEdge([NodeIdx(0), NodeIdx(9)]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert_eq!(faces, faces_exp);

        let edge = DirectedEdge([NodeIdx(9), NodeIdx(0)]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert_eq!(faces, faces_exp);
        
        // Edge 7 -> 6 and 6 -> 7
        let faces_exp = HashSet::from([&msh.elems[2]]);
        let edge = DirectedEdge([NodeIdx(7), NodeIdx(6)]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert_eq!(faces, faces_exp);

        let edge = DirectedEdge([NodeIdx(6), NodeIdx(7)]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert_eq!(faces, faces_exp);
        
        // Edge 1 -> 2 and 2 -> 1
        let faces_exp = HashSet::from([&msh.elems[0]]);
        let edge = DirectedEdge([NodeIdx(1), NodeIdx(2)]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert_eq!(faces, faces_exp);

        let edge = DirectedEdge([NodeIdx(2), NodeIdx(1)]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert_eq!(faces, faces_exp);
        
        // Edge 4 -> 0 (non existent)
        let edge = DirectedEdge([NodeIdx(4), NodeIdx(0)]);
        let faces: HashSet<&QuadNodes> = msh.faces_of_edge(edge).collect();
        assert!(faces.is_empty());
    }
}