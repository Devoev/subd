use crate::cells::chain::Chain;

use crate::cells::node::Node;
use crate::cells::traits::{Cell, CellBoundary, CellConnectivity, OrderedCell, OrientedCell, ToElement};
use crate::element::line_segment::LineSegment;
use crate::mesh::traits::VertexStorage;
use nalgebra::{clamp, Const, DimName, DimNameSub, Point, RealField, U1};
use std::cmp::minmax;
use std::hash::Hash;

/// A *directed* edge between two nodes.
/// The topological structure is
/// ```text
///    0 --- 1
/// -+---> u
/// ```
/// where `0` is the start and `1` the end node.
/// Geometrically the pair defines the topology of a [`LineSegment`].
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct DirectedEdge(pub [Node; 2]);

impl DirectedEdge {
    /// Returns the start node of this edge.
    pub fn start(&self) -> Node {
        self.0[0]
    }

    /// Returns the end node of this edge.
    pub fn end(&self) -> Node {
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

impl Cell for DirectedEdge {
    type Dim = U1;
    type Node = Node;

    fn nodes(&self) -> &[Self::Node] {
        &self.0.map(|node| node.0)
    }
}

impl <T: RealField + Copy, const M: usize> ToElement<T, Const<M>> for DirectedEdge {
    type Elem = LineSegment<T, M>;
    type Coords = Vec<Point<T, M>>;

    fn to_element(&self, coords: &Self::Coords) -> Self::Elem {
        LineSegment::new(self.0.map(|node| coords.vertex(node.0)))
    }
}

impl CellConnectivity for DirectedEdge {
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

impl CellBoundary for DirectedEdge {
    const NUM_SUB_CELLS: usize = 2;
    type SubCell = Node;
    type Boundary = NodePair;

    fn boundary(&self) -> Self::Boundary {
        NodePair(self.0)
    }
}

impl OrderedCell for DirectedEdge {
    fn sorted(&self) -> Self {
        DirectedEdge(minmax(self.start(), self.end()))
    }
}

impl OrientedCell for DirectedEdge {
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
pub struct NodePair(pub [Node; 2]);

impl Chain<Node> for NodePair {
    fn cells(&self) -> &[Node] {
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
    sorted_nodes: [Node; 2]
}

impl UndirectedEdge {
    /// Constructs a new [`UndirectedEdge`] from the given nodes `a` and `b`
    /// by sorting the nodes such that `start < end`.
    pub fn new(a: Node, b: Node) -> Self {
        // todo: test for a == b
        UndirectedEdge { sorted_nodes: minmax(a, b) }
    }

    /// Attempts to construct a new [`UndirectedEdge`] from the given `start` and `end` nodes.
    /// If `start >= end` `None` is returned.
    pub fn try_new(start: Node, end: Node) -> Option<Self> {
        (start < end).then_some(UndirectedEdge { sorted_nodes: [start, end] })
    }

    /// Returns the array of sorted nodes.
    pub fn sorted_nodes(&self) -> &[Node; 2] {
        &self.sorted_nodes
    }

    /// Returns the node with the lower index.
    pub fn first(&self) -> Node {
        self.sorted_nodes[0]
    }

    /// Returns the node with the greater index.
    pub fn second(&self) -> Node {
        self.sorted_nodes[1]
    }
}

impl From<DirectedEdge> for UndirectedEdge {
    /// Turns a directed edge into an undirected one, by removing the orientation.
    fn from(value: DirectedEdge) -> Self {
        UndirectedEdge::new(value.start(), value.end())
    }
}

impl From<UndirectedEdge> for DirectedEdge {
    /// Turns an undirected edge into a directed one,
    /// by choosing the orientation such that `start < end`.
    fn from(value: UndirectedEdge) -> Self {
        DirectedEdge(value.sorted_nodes)
    }
}

impl Cell for UndirectedEdge {
    type Dim = U1;
    type Node = Node;

    fn nodes(&self) -> &[Self::Node] {
        &self.sorted_nodes.map(|node| node.0)
    }
}

impl CellConnectivity for UndirectedEdge {
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