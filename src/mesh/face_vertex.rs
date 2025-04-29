//! Data structures for a [face-vertex mesh](https://en.wikipedia.org/wiki/Polygon_mesh#Face-vertex_meshes).

use std::cmp::minmax;
use itertools::Itertools;
use nalgebra::{Point2, RealField, Vector2};
use crate::subd::patch::Patch;

/// Index of a node in the mesh.
pub type NodeIdx = usize;

/// Topology of an edge. The topology is defined as
/// ```text
///    0 --- 1
/// -+---> u
/// ```
/// where `0` is the start and `1` the end node.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct EdgeTopo(pub [NodeIdx; 2]);

impl EdgeTopo {

    /// Returns the start node of this edge.
    pub fn start(&self) -> NodeIdx {
        self.0[0]
    }

    /// Returns the end node of this edge.
    pub fn end(&self) -> NodeIdx {
        self.0[1]
    }

    /// Returns the boundary of this edge, i.e. the starting and end nodes.
    pub fn boundary(&self) -> [NodeIdx; 2] {
        self.0
    }

    /// Returns a sorted copy of this edge such that `self.start() < self.end()`.
    pub fn sorted(&self) -> EdgeTopo {
        EdgeTopo(minmax(self.start(), self.end()))
    }
    
    /// Changes the orientation of this edge by calling [`EdgeTopo::sorted`] on self.
    pub fn sort(&mut self) {
        *self = self.sorted();
    }

    /// Returns a copy of this edge with reversed orientation.
    pub fn reversed(&self) -> EdgeTopo {
        EdgeTopo([self.end(), self.start()])
    }

    /// Reversed the orientation of this edge by calling [`EdgeTopo::reversed`].
    pub fn reverse(&mut self) {
        *self = self.reversed();
    }
}

/// Topology of a 2D quadrilateral face. The topology is defined as
/// ```text
/// v 3 --- 2
/// ^ |     |
/// | 0 --- 1
/// +---> u
/// ```
/// where `0,1,2,3` are the corner nodes of the face.
#[derive(Debug, Clone, Copy)]
pub struct QuadTopo2d([NodeIdx; 4]);

impl QuadTopo2d {
    /// Returns the corner nodes.
    pub fn nodes(&self) -> [NodeIdx; 4] {
        self.0
    }

    // todo: add boundary function with return value of connected edges, i.e. a chain topology
    /// Returns all 4 edges of this quadrilateral face in the following order
    /// ```text
    ///   + -- 2 -- +
    ///   |         |
    /// v 3         1
    /// ^ |         |
    /// | + -- 0 -- +
    /// +---> u
    /// ```
    pub fn edges(&self) -> [EdgeTopo; 4] {
        let [a, b, c, d] = self.0;
        [EdgeTopo([a, b]), EdgeTopo([b, c]), EdgeTopo([c, d]), EdgeTopo([d, a])]
    }

    // todo: return an intersection result (possibly an enum)
    /// Returns the intersection between `self` and `other` as an iterator of the overlapping nodes.
    pub fn intersection(&self, other: QuadTopo2d) -> impl Iterator<Item=NodeIdx> {
        self.nodes().into_iter().filter(move |n| other.nodes().contains(n))
    }

    /// Returns whether `self` and `other` are adjacent, i.e. share an edge.
    pub fn is_adjacent(&self, other: QuadTopo2d) -> bool {
        self.intersection(other).count() == 2
    }

    /// Returns whether `self` and `other` are touching, i.e. share an edge or a node.
    pub fn is_touching(&self, other: QuadTopo2d) -> bool {
        let count = self.intersection(other).count();
        count == 2 || count == 1
    }

    /// Returns a sorted copy of this face,
    /// such that the given `node` is at the local node position `local_idx`.
    ///
    /// # Example
    /// For `local_idx=3` the faces nodes get sorted as
    /// ```text
    /// v 3 --- 2         n --- 0
    /// ^ |     |   ==>   |     |
    /// | 0 --- n         2 --- 3
    /// +---> u
    /// ```
    /// i.e. the given node `n` moves from the original position `1`
    /// to position `local_idx=3`.
    pub fn sorted_by_node(&self, node: NodeIdx, local_idx: usize) -> QuadTopo2d {
        let original_idx = self.nodes().iter().position(|&n| n == node).unwrap();
        let mut nodes = self.nodes();
        if local_idx > original_idx {
            nodes.rotate_right(local_idx - original_idx);
        } else {
            nodes.rotate_left(original_idx - local_idx);
        }
        QuadTopo2d(nodes)
    }

    /// Returns a sorted copy of this face, such that the node `uv_origin` is the first node.
    /// Assumes the face is initially sorted in positive orientation.
    ///
    /// # Example
    /// If the node `uv_origin` is the second node in the faces local sorting,
    /// i.e. at the local index `1`, the nodes get sorted as
    /// ```text
    /// v 2 --- 1         3 --- 2
    /// ^ |     |   ==>   |     |
    /// | 3 --- 0         0 --- 1
    /// +---> u
    /// ```
    /// where `0` is the `uv_origin`.
    pub fn sorted_by_origin(&self, uv_origin: NodeIdx) -> QuadTopo2d {
        self.sorted_by_node(uv_origin, 0)
    }
}

/// Topology of a 2D quadrilateral face-vertex mesh.
pub struct QuadMeshTopo2d {
    /// Face connectivity vector.
    pub faces: Vec<QuadTopo2d>
}

impl QuadMeshTopo2d {
    /// Returns an iterator over all unique and sorted edges in this mesh.
    pub fn edges(&self) -> impl Iterator<Item = EdgeTopo> + '_ {
        self.faces.iter()
            .flat_map(|&face| face.edges())
            .map(|edge| edge.sorted())
            .unique()
    }

    /// Returns all edges connected to the given `node`.
    pub fn edges_of_node(&self, node: NodeIdx) -> impl Iterator<Item = EdgeTopo> + '_ {
        self.edges().filter(move |edge| edge.0.contains(&node))
    }

    /// Calculates the valence of the given `node`, i.e. the number of edges connected to the node.
    pub fn valence(&self, node: NodeIdx) -> usize {
        self.edges_of_node(node).count()
    }

    /// Returns all `(index,face)`-pairs of faces who have the given `node` as a vertex.
    pub fn faces_of_node(&self, node: NodeIdx) -> impl Iterator<Item = (usize, &QuadTopo2d)> {
        self.faces
            .iter()
            .enumerate()
            .filter(move |(_, face)| face.nodes().contains(&node))
    }

    /// Returns `true` if the face is regular.
    pub fn is_regular(&self, face: QuadTopo2d) -> bool {
        face.nodes().iter().all(|node| self.valence(*node) == 4)
    }

    /// Finds the irregular node of the given `face`, if any exists.
    pub fn irregular_node_of_face(&self, face: QuadTopo2d) -> Option<NodeIdx> {
        face.nodes()
            .into_iter()
            .find(|&v| self.valence(v) != 4)
    }

    /// Returns all adjacent faces to `face`.
    pub fn adjacent_faces(&self, face: QuadTopo2d) -> impl Iterator<Item = (usize, &QuadTopo2d)> {
        self.faces
            .iter()
            .enumerate()
            .filter(move |(_, f)| f.is_adjacent(face))
    }

    /// Returns whether the given `face` is a boundary face, i.e. it has less than `4` adjacent faces.
    pub fn is_boundary_face(&self, face: QuadTopo2d) -> bool {
        self.adjacent_faces(face).count() < 4
    }

    /// Returns whether the given `node` is a boundary node,
    /// i.e. all faces containing the node are boundary faces.
    pub fn is_boundary_node(&self, node: NodeIdx) -> bool {
        self.faces_of_node(node).all(|(_, &f)| self.is_boundary_face(f))
    }

    /// Finds all boundary nodes of the given `face`,
    /// i.e. all irregular nodes, assuming the face is a boundary face.
    pub fn boundary_nodes_of_face(&self, face: QuadTopo2d) -> Vec<NodeIdx> {
        face.nodes().into_iter().filter(|&v| self.valence(v) != 4).collect()
    }
}

/// A 2D quadrilateral face-vertex mesh with geometric data of the coordinates of each vertex.
pub struct QuadMesh2d<T: RealField> {
    /// Coordinates of the meshes vertices.
    pub coords: Vec<Point2<T>>,
    /// Topological connectivity of the faces.
    pub topology: QuadMeshTopo2d,
}

impl<T: RealField> QuadMesh2d<T> {
    /// Returns the [`Point2`] of the given `node` index.
    pub fn coords(&self, node: NodeIdx) -> &Point2<T> {
        &self.coords[node]
    }

    // todo: possibly move this method to topology
    /// Returns the number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.coords.len()
    }

    /// Returns the nodes of the given edge topology `edge_top`.
    pub fn coords_of_edge(&self, edge_top: EdgeTopo) -> [&Point2<T>; 2] {
        edge_top.0.map(|n| self.coords(n))
    }

    /// Returns the nodes of the given face topology `face_top`.
    pub fn coords_of_face(&self, face_top: QuadTopo2d) -> [&Point2<T>; 4] {
        face_top.0.map(|node| self.coords(node))
    }

    /// Computes the centroid of the given face topology `face_top`.
    pub fn centroid(&self, face_top: QuadTopo2d) -> Point2<T> {
        let corners = self.coords_of_face(face_top);
        let centroid = corners
            .iter()
            .map(|p| &p.coords)
            .sum::<Vector2<T>>() / T::from_f64(4.0).unwrap();
        Point2::from(centroid)
    }

    // todo: possibly move this method to topology
    /// Returns an iterator over the indices of all boundary nodes in this mesh.
    pub fn boundary_nodes(&self) -> impl Iterator<Item = NodeIdx> + '_ {
        (0..self.num_nodes()).filter(|&n| self.topology.is_boundary_node(n))
    }

    /// Finds the patch of the regular or irregular `face`.
    pub fn find_patch(&self, face: QuadTopo2d) -> Patch<T> {
        todo!()
    }

    /// Returns an iterator over all patches in this mesh.
    pub fn patches(&self) -> impl Iterator<Item =Patch<T>> {
        self.topology.faces.iter().map(|&face| self.find_patch(face))
    }
}