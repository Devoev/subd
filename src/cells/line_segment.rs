use crate::cells::chain::Chain;
use crate::cells::node::NodeIdx;
use crate::cells::topo::{Cell, CellBoundary, OrderedCell, OrientedCell};
use crate::mesh::face_vertex::QuadVertexMesh;
use nalgebra::{clamp, DimName, DimNameSub, Point, RealField, U0, U1};
use std::cmp::minmax;
use crate::cells::geo;
use crate::cells::lerp::Lerp;
use crate::cells::unit_cube::UnitCube;

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
    pub fn from_msh(topology: NodePair, msh: &QuadVertexMesh<T, M>) -> Self {
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

/// Pair of 2 nodes, defining an *edge*.
/// The topological structure is
/// ```text
///    0 --- 1
/// -+---> u
/// ```
/// where `0` is the start and `1` the end node.
/// Geometrically the pair defines the topology of a [`LineSegment`].
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct NodePair(pub [NodeIdx; 2]);

impl NodePair {
    /// Returns the start node of this edge.
    pub fn start(&self) -> NodeIdx {
        self.0[0]
    }

    /// Returns the end node of this edge.
    pub fn end(&self) -> NodeIdx {
        self.0[1]
    }

    /// Returns a sorted copy of this edge such that `self.start() < self.end()`.
    pub fn sorted(&self) -> NodePair {
        NodePair(minmax(self.start(), self.end()))
    }

    /// Changes the orientation of this edge by calling [`NodePair::sorted`] on self.
    pub fn sort(&mut self) {
        *self = self.sorted();
    }

    /// Returns a copy of this edge with reversed orientation.
    pub fn reversed(&self) -> NodePair {
        NodePair([self.end(), self.start()])
    }

    /// Reverses the orientation of this edge by calling [`NodePair::reversed`].
    pub fn reverse(&mut self) {
        *self = self.reversed();
    }
}

impl Cell<U1> for NodePair {
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

impl CellBoundary<U1> for NodePair {
    const NUM_SUB_CELLS: usize = 2;
    type SubCell = NodeIdx;
    type Boundary = LineSegmentBndTopo;

    fn boundary(&self) -> Self::Boundary {
        LineSegmentBndTopo(self.0)
    }
}

impl OrderedCell<U1> for NodePair {
    fn sorted(&self) -> Self {
        NodePair(minmax(self.start(), self.end()))
    }
}

impl OrientedCell<U1> for NodePair {
    fn orientation(&self) -> i8 {
        clamp(self.end().0 as i8 - self.start().0 as i8, -1, 1)
    }

    fn reversed(&self) -> Self {
        NodePair([self.end(), self.start()])
    }
}

pub struct LineSegmentBndTopo(pub [NodeIdx; 2]);

impl Chain<U0, NodeIdx> for LineSegmentBndTopo {
    fn cells(&self) -> &[NodeIdx] {
        &self.0
    }
}