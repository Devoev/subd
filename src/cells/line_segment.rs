use crate::cells::chain::Chain;
use crate::cells::geo;
use crate::cells::lerp::Lerp;
use crate::cells::node::NodeIdx;
use crate::cells::topo::{Cell, CellBoundary, OrderedCell, OrientedCell};
use crate::cells::unit_cube::UnitCube;
use crate::mesh::face_vertex::QuadVertexMesh;
use nalgebra::{clamp, DimName, DimNameSub, Point, RealField, U0, U1};
use std::cmp::minmax;
use std::hash::Hash;

/// A line segment, i.e. a straight line bounded by 2 points
/// in [`M`]-dimensional space.
pub struct LineSegment<T: RealField, const M: usize> {
    pub vertices: [Point<T, M>; 2]
}

impl<T: RealField, const M: usize> LineSegment<T, M> {

    /// Constructs a new [`LineSegment`] from the given `vertices`.
    pub fn new(vertices: [Point<T, M>; 2]) -> Self {
        LineSegment { vertices }
    }

    /// Constructs a new [`LineSegment`] from the given `topology` and `msh`.
    pub fn from_msh(topology: DirectedEdge, msh: &QuadVertexMesh<T, M>) -> Self {
        LineSegment::new(topology.0.map(|n| msh.coords(n).clone()))
    }
}

impl <T: RealField + Copy, const M: usize> geo::Cell<T, T, 1, M> for LineSegment<T, M> {
    type RefCell = UnitCube<1>;
    type GeoMap = Lerp<T, M>;

    fn ref_cell(&self) -> Self::RefCell {
        UnitCube
    }

    fn geo_map(&self) -> Self::GeoMap {
        Lerp::new(self.vertices[0], self.vertices[1])
    }
}

/// A *directed* edge between two nodes.
/// The topological structure is
/// ```text
///    0 --- 1
/// -+---> u
/// ```
/// where `0` is the start and `1` the end node.
/// Geometrically the pair defines the topology of a [`LineSegment`].
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct DirectedEdge(pub [NodeIdx; 2]);

impl DirectedEdge {
    /// Returns the start node of this edge.
    pub fn start(&self) -> NodeIdx {
        self.0[0]
    }

    /// Returns the end node of this edge.
    pub fn end(&self) -> NodeIdx {
        self.0[1]
    }

    /// Returns a sorted copy of this edge such that `self.start() < self.end()`.
    pub fn sorted(&self) -> DirectedEdge {
        DirectedEdge(minmax(self.start(), self.end()))
    }

    /// Changes the orientation of this edge by calling [`DirectedEdge::sorted`] on self.
    pub fn sort(&mut self) {
        *self = self.sorted();
    }

    /// Returns a copy of this edge with reversed orientation.
    pub fn reversed(&self) -> DirectedEdge {
        DirectedEdge([self.end(), self.start()])
    }

    /// Reverses the orientation of this edge by calling [`DirectedEdge::reversed`].
    pub fn reverse(&mut self) {
        *self = self.reversed();
    }
}

impl Cell<U1> for DirectedEdge {
    fn nodes(&self) -> &[NodeIdx] {
        &self.0
    }

    fn is_connected<M: DimName>(&self, other: &Self, dim: M) -> bool
    where
        U1: DimNameSub<M>
    {
        match dim.value() {
            1 => { // edges are the same
                self.start() == other.start() && self.end() == other.end()
                    || self.start() == other.end() && self.end() == other.start()
            },
            0 => { // edges share a node
                self.start() == other.start()
                    || self.start() == other.end()
                    || self.end() == other.start()
                    || self.end() == other.end()
            },
            _ => unreachable!("Dimension `M` (is {dim:?}) should be <= `K` (is 1)")
        }
    }
}

impl CellBoundary<U1> for DirectedEdge {
    const NUM_SUB_CELLS: usize = 2;
    type SubCell = NodeIdx;
    type Boundary = NodePair;

    fn boundary(&self) -> Self::Boundary {
        NodePair(self.0)
    }
}

impl OrderedCell<U1> for DirectedEdge {
    fn sorted(&self) -> Self {
        DirectedEdge(minmax(self.start(), self.end()))
    }
}

impl OrientedCell<U1> for DirectedEdge {
    fn orientation(&self) -> i8 {
        clamp(self.end().0 as i8 - self.start().0 as i8, -1, 1)
    }

    fn orientation_eq(&self, other: &Self) -> bool {
        self.start() == other.start() && self.end() == other.end()
    }

    fn reversed(&self) -> Self {
        DirectedEdge([self.end(), self.start()])
    }
}

/// Pair of two disjoint nodes.
pub struct NodePair(pub [NodeIdx; 2]);

impl Chain<U0, NodeIdx> for NodePair {
    fn cells(&self) -> &[NodeIdx] {
        &self.0
    }
}

/// An *undirected* edge between two nodes.
/// The topological and geometric structure is essentially the same as [`DirectedEdge`],
/// but the ordering of nodes doesn't matter.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct UndirectedEdge {
    /// The sorted nodes defining the start and end of the edge.
    /// The sorting is such that `start < end`.
    sorted_nodes: [NodeIdx; 2]
}

impl UndirectedEdge {
    /// Constructs a new [`UndirectedEdge`] from the given nodes `a` and `b`
    /// by sorting the nodes such that `start < end`.
    pub fn new(a: NodeIdx, b: NodeIdx) -> Self {
        // todo: test for a == b
        UndirectedEdge { sorted_nodes: minmax(a, b) }
    }

    /// Attempts to construct a new [`UndirectedEdge`] from the given `start` and `end` nodes.
    /// If `start >= end` `None` is returned.
    pub fn try_new(start: NodeIdx, end: NodeIdx) -> Option<Self> {
        (start < end).then_some(UndirectedEdge { sorted_nodes: [start, end] })
    }

    /// Returns the array of sorted nodes.
    pub fn sorted_nodes(&self) -> &[NodeIdx; 2] {
        &self.sorted_nodes
    }

    /// Returns the node with the lower index.
    pub fn first(&self) -> NodeIdx {
        self.sorted_nodes[0]
    }

    /// Returns the node with the greater index.
    pub fn second(&self) -> NodeIdx {
        self.sorted_nodes[1]
    }
}

impl From<DirectedEdge> for UndirectedEdge {
    fn from(value: DirectedEdge) -> Self {
        UndirectedEdge::new(value.start(), value.end())
    }
}

impl Cell<U1> for UndirectedEdge {
    fn nodes(&self) -> &[NodeIdx] {
        &self.sorted_nodes
    }

    fn is_connected<M: DimName>(&self, other: &Self, dim: M) -> bool
    where
        U1: DimNameSub<M>
    {
        match dim.value() {
            1 => { // edges are the same
                self == other
            },
            0 => { // edges share a node
                self.first() == other.first()
                    || self.first() == other.second()
                    || self.second() == other.first()
                    || self.second() == other.second()
            },
            _ => unreachable!("Dimension `M` (is {dim:?}) should be <= `K` (is 1)")
        }
    }
}