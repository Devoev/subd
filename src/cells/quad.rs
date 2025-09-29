use std::iter::zip;
use itertools::Itertools;
use crate::cells::chain::{Chain, ChainBoundary};
use crate::cells::geo;
use crate::cells::lerp::BiLerp;
use crate::cells::line_segment::{DirectedEdge, UndirectedEdge};
use crate::cells::node::NodeIdx;
use crate::cells::topo::{CellToNodes, CellBoundary, OrientedCell, Cell};
use crate::cells::unit_cube::UnitCube;
use crate::mesh::face_vertex::QuadVertexMesh;
use nalgebra::{Const, DefaultAllocator, DimName, DimNameSub, Point, RealField, SVector, Scalar, U1, U2};
use nalgebra::allocator::Allocator;
use crate::mesh::elem_vertex::{ElemVec, ElemToVertexMesh};
use crate::mesh::traits::{Mesh, VertexStorage};

/// A 2d quadrilateral element of topology [`QuadNodes`],
/// embedded in [`M`]-dimensional space.
#[derive(Debug, Clone, Copy)]
pub struct Quad<T: RealField, const M: usize> {
    pub vertices: [Point<T, M>; 4]
}

impl<T: RealField, const M: usize> Quad<T, M> {

    /// Constructs a new [`Quad`] from the given `vertices`.
    pub fn new(vertices: [Point<T, M>; 4]) -> Self {
        Quad { vertices }
    }

    /// Constructs a new [`Quad`] from the given `topology` and `msh`.
    pub fn from_msh(topology: QuadNodes, msh: &QuadVertexMesh<T, M>) -> Self {
        Quad::new(topology.0.map(|n| msh.coords(n).clone()))
    }

    /// Computes the centroid of this face.
    pub fn centroid(&self) -> Point<T, M> {
        let centroid = self.vertices
            .iter()
            .map(|p| &p.coords)
            .sum::<SVector<T, M>>() / T::from_f64(4.0).unwrap();
        Point::from(centroid)
    }
}

impl <T: RealField + Copy, const M: usize> geo::Cell<T> for Quad<T, M> {
    type ParametricCell = UnitCube<2>;
    type GeoMap = BiLerp<T, M>;

    fn ref_cell(&self) -> Self::ParametricCell {
        UnitCube
    }

    fn geo_map(&self) -> Self::GeoMap {
        BiLerp::new(self.vertices)
    }
}


/// Face-to-nodes topology of a 2D quadrilateral given by
/// ```text
/// v 3 --- 2
/// ^ |     |
/// | 0 --- 1
/// +---> u
/// ```
/// where `0,1,2,3` are the four corner nodes of the quadrilateral.
#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct QuadNodes(pub [NodeIdx; 4]);

impl QuadNodes {
    /// Constructs a new [`QuadNodes`] from the given indices `i,j,k,l` of the corner nodes.
    pub fn from_indices(i: usize, j: usize, k: usize, l: usize) -> Self {
        QuadNodes([NodeIdx(i), NodeIdx(j), NodeIdx(k), NodeIdx(l)])
    }
    /// Returns the corner nodes.
    pub fn nodes(&self) -> [NodeIdx; 4] {
        self.0
    }

    /// Returns all 4 *directed* edges of this quadrilateral face in the following order
    /// ```text
    ///   ← -- 2 -- ↑
    ///   |         |
    /// v 3         1
    /// ^ |         |
    /// | ↓ -- 0 -- →
    /// +---> u
    /// ```
    /// where the arrows indicate orientation of edges.
    pub fn edges(&self) -> [DirectedEdge; 4] {
        let [a, b, c, d] = self.0;
        [DirectedEdge([a, b]), DirectedEdge([b, c]), DirectedEdge([c, d]), DirectedEdge([d, a])]
    }

    /// Returns all 4 *undirected* edges of this quadrilateral face in the following order
    /// ```text
    ///   + -- 2 -- +
    ///   |         |
    /// v 3         1
    /// ^ |         |
    /// | + -- 0 -- +
    /// +---> u
    /// ```
    pub fn undirected_edges(&self) -> [UndirectedEdge; 4] {
        self.edges().map(UndirectedEdge::from)
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
    pub fn sorted_by_node(&self, node: NodeIdx, local_idx: usize) -> QuadNodes {
        let original_idx = self.nodes().iter().position(|&n| n == node).unwrap();
        let mut nodes = self.nodes();
        if local_idx > original_idx {
            nodes.rotate_right(local_idx - original_idx);
        } else {
            nodes.rotate_left(original_idx - local_idx);
        }
        QuadNodes(nodes)
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
    pub fn sorted_by_origin(&self, uv_origin: NodeIdx) -> QuadNodes {
        self.sorted_by_node(uv_origin, 0)
    }
}

/// Common nodes of two quadrilaterals.
pub enum QuadNodesIntersection {
    /// No common nodes.
    Empty,

    /// The single common node.
    Node(NodeIdx),

    /// The two common, connected nodes of an edge.
    Edge(NodeIdx, NodeIdx),

    /// The four common nodes, i.e. the same face.
    Face(NodeIdx, NodeIdx, NodeIdx, NodeIdx),
}

impl QuadNodesIntersection {
    /// Constructs a new [`QuadNodesIntersection`] from the given two quads `q1` and `q2`.
    pub fn new(q1: &QuadNodes, q2: &QuadNodes) -> Self {
        let common_nodes = q1.nodes()
            .into_iter()
            .filter(move |n| q2.nodes().contains(n))
            .collect_vec();

        match common_nodes[..] {
            [] => QuadNodesIntersection::Empty,
            [a] => QuadNodesIntersection::Node(a),
            [a, b] => QuadNodesIntersection::Edge(a, b),
            [a, b, c, d] => QuadNodesIntersection::Face(a, b, c, d),
            _ => unreachable!("It is impossible that two quadrilateral faces share only 3 nodes, or more than 4 nodes."),
        }
    }

    /// Returns the (first) shared node of the two quads `q1` and `q2`.
    /// If they don't share a node, `None` is returned.
    pub fn new_shared_node(q1: &QuadNodes, q2: &QuadNodes) -> Option<NodeIdx> {
        let nodes1 = q1.nodes();
        let nodes2 = q2.nodes();
        nodes1.into_iter().find(|n| nodes2.contains(n))
    }

    /// Returns the shared edge of the two quads `q1` and `q2`.
    /// The edge's orientation is determined by `q1`.
    /// If they don't share an edge, `None` is returned.
    pub fn new_shared_edge(q1: &QuadNodes, q2: &QuadNodes) -> Option<DirectedEdge> {
        let edges1 = q1.edges();
        let edges2 = q2.edges();
        edges1.into_iter().find(|edge| edges2.contains(&edge.reversed()))
    }
}

impl QuadNodes {
    /// Returns the [`QuadNodesIntersection`] between `self` and `other`..
    fn intersection(&self, other: QuadNodes) -> QuadNodesIntersection {
        QuadNodesIntersection::new(self, &other)
    }

    /// Returns whether `self` and `other` are adjacent, i.e. share an edge.
    pub fn is_adjacent(&self, other: QuadNodes) -> bool {
        matches!(self.intersection(other), QuadNodesIntersection::Edge(_, _))
    }

    /// Returns whether `self` and `other` are touching, i.e. share an edge or a node.
    pub fn is_touching(&self, other: QuadNodes) -> bool {
        matches!(self.intersection(other), QuadNodesIntersection::Node(_) | QuadNodesIntersection::Edge(_, _))
    }

    /// Returns the directed edge shared by both `self` and `other`.
    /// If they don't share an edge, `None` is returned.
    /// The edges orientation is determined by `self`.
    pub fn shared_edge(&self, other: QuadNodes) -> Option<DirectedEdge> {
        QuadNodesIntersection::new_shared_edge(self, &other)
    }

    /// Returns the (first) node shared by both `self` and `other`.
    /// If they don't share a node, `None` is returned.
    pub fn shared_node(&self, other: QuadNodes) -> Option<NodeIdx> {
        QuadNodesIntersection::new_shared_node(self, &other)
    }
}

impl <T: RealField, const M: usize> Cell<T, Const<M>> for QuadNodes {
    type GeoCell = Quad<T, M>;
    type Coords = Vec<Point<T, M>>;

    fn to_geo_cell(&self, coords: &Self::Coords) -> Self::GeoCell {
        Quad::new(self.0.map(|node| coords.vertex(node.0)))
    }
}

impl CellToNodes for QuadNodes {
    type Dim = U2;

    fn nodes(&self) -> &[NodeIdx] {
        &self.0
    }

    fn is_connected<M: DimName>(&self, other: &Self, dim: M) -> bool
    where
        U2: DimNameSub<M>
    {
        let num_shared_nodes = self.nodes()
            .into_iter()
            .filter(|n| other.nodes().contains(n))
            .count();

        match dim.value() {
            2 => num_shared_nodes == 4, // faces are the same
            1 => num_shared_nodes == 2, // faces share an edge
            0 => num_shared_nodes == 1, // faces share a node
            _ => unreachable!("Dimension `M` (is {dim:?}) should be <= `K` (is 2)"),
        }
    }
}

impl OrientedCell for QuadNodes {
    fn orientation(&self) -> i8 {
        todo!("possible orientation:\
        + smallest node -> 2nd smallest node \
        - smallest node -> 2nd largest or largest node \
        ")
    }

    fn orientation_eq(&self, other: &Self) -> bool {
        if !self.topo_eq(other) { return false; }
        let other = other.sorted_by_origin(self.0[0]);
        zip(self.edges(), other.edges())
            .all(|(ei, ej)| ei.orientation_eq(&ej))
    }

    fn reversed(&self) -> Self {
        let mut nodes = self.nodes();
        nodes.reverse();
        QuadNodes(nodes)
    }
}

impl CellBoundary for QuadNodes {
    const NUM_SUB_CELLS: usize = 4;
    type SubCell = DirectedEdge;
    type Boundary = QuadBndTopo;

    fn boundary(&self) -> Self::Boundary {
        QuadBndTopo(self.edges())
    }
}

pub struct QuadBndTopo(pub [DirectedEdge; 4]);

impl Chain<DirectedEdge> for QuadBndTopo {
    fn cells(&self) -> &[DirectedEdge] {
        &self.0
    }
}