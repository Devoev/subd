//! Special cases of an [`ElemVertexMesh`] for `K = 2`, i.e. where the elements are faces.

use crate::cells::chain::Chain;
use crate::cells::line_segment::LineSegment;
use crate::cells::node::NodeIdx;
use crate::cells::quad::{Quad, QuadNodes};
use crate::cells::topo::{Cell, CellBoundary, Edge2, OrderedCell};
use crate::mesh::elem_vertex::ElemVertexMesh;
use itertools::Itertools;
use nalgebra::{RealField, U1, U2};
use std::hash::Hash;

/// A face-vertex mesh with `2`-dimensional faces [`C`].
pub type FaceVertexMesh<T, C, const M: usize> = ElemVertexMesh<T, C, 2, M>;

/// A face-vertex mesh with quadrilateral faces.
pub type QuadVertexMesh<T, const M: usize> = FaceVertexMesh<T, QuadNodes, M>;

impl <T: RealField, F: CellBoundary<U2>, const M: usize> FaceVertexMesh<T, F, M>
where Edge2<F>: OrderedCell<U1> + Clone + Eq + Hash
{
    /// Returns an iterator over all unique and sorted edges in this mesh.
    pub fn edges(&self) -> impl Iterator<Item = Edge2<F>> + '_ {
        self.elems.iter()
            .flat_map(|face| face.boundary().cells().to_owned())
            .map(|edge: Edge2<F> | edge.sorted())
            .unique()
    }

    /// Returns all edges connected to the given `node`.
    pub fn edges_of_node(&self, node: NodeIdx) -> impl Iterator<Item = Edge2<F>> + '_ {
        self.edges().filter(move |edge| edge.contains_node(node))
    }

    /// Calculates the valence of the given `node`, i.e. the number of edges connected to the node.
    pub fn valence(&self, node: NodeIdx) -> usize {
        self.edges_of_node(node).count()
    }
}

impl<T: RealField, const M: usize> QuadVertexMesh<T, M> {
    /// Returns `true` if the face is regular.
    pub fn is_regular(&self, face: QuadNodes) -> bool {
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