//! Data structures for a [face-vertex mesh](https://en.wikipedia.org/wiki/Polygon_mesh#Face-vertex_meshes).

use std::cmp::minmax;
use nalgebra::{Point2, RealField};

/// Index of a node in the mesh.
pub type NodeIdx = usize;

/// Topology of an edge. The topology is defined as
/// ```text
///    0 --- 1
/// -+---> u
/// ```
/// where `0` is the start and `1` the end node.
#[derive(Debug, Copy, Clone)]
pub struct EdgeTopo([NodeIdx; 2]);

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

    // todo: update doc and change to connected edges, i.e. a chain topology
    /// Returns the boundary of this quadrilateral face, i.e. all 4 edges in the following order
    /// ```text
    ///   + -- 2 -- +
    ///   |         |
    /// v 3         1
    /// ^ |         |
    /// | + -- 0 -- +
    /// +---> u
    /// ```
    pub fn boundary(&self) -> [EdgeTopo; 4] {
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

/// A 2D quadrilateral face-vertex mesh with geometric data of the coordinates of each vertex.
pub struct QuadMesh2d<T: RealField> {
    /// Coordinates of the meshes vertices.
    pub coords: Vec<Point2<T>>,
    /// Topological connectivity of the faces.
    pub topology: QuadMeshTopo2d,
}