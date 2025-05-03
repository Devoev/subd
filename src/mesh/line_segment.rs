use std::cmp::minmax;
use nalgebra::{Point2, RealField};
use crate::mesh::face_vertex::{NodeIdx, QuadVertexMesh};

/// A line segment of topology [`LineSegmentTopo`].
pub struct LineSegment<T: RealField> {
    pub vertices: [Point2<T>; 2]
}

impl<T: RealField> LineSegment<T> {

    /// Constructs a new [`LineSegment`] from the given `vertices`.
    pub fn new(vertices: [Point2<T>; 2]) -> Self {
        LineSegment { vertices }
    }

    /// Constructs a new [`LineSegment`] from the given `topology` and `msh`.
    pub fn from_msh(topology: LineSegmentTopo, msh: &QuadVertexMesh<T>) -> Self {
        LineSegment::new(topology.0.map(|n| msh.coords(n).clone()))
    }
}

/// Topology of a line segment, i.e. a straight line bounded by 2 points. The topology is defined as
/// ```text
///    0 --- 1
/// -+---> u
/// ```
/// where `0` is the start and `1` the end node.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct LineSegmentTopo(pub [NodeIdx; 2]);

impl LineSegmentTopo {

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
    pub fn sorted(&self) -> LineSegmentTopo {
        LineSegmentTopo(minmax(self.start(), self.end()))
    }

    /// Changes the orientation of this edge by calling [`LineSegmentTopo::sorted`] on self.
    pub fn sort(&mut self) {
        *self = self.sorted();
    }

    /// Returns a copy of this edge with reversed orientation.
    pub fn reversed(&self) -> LineSegmentTopo {
        LineSegmentTopo([self.end(), self.start()])
    }

    /// Reverses the orientation of this edge by calling [`LineSegmentTopo::reversed`].
    pub fn reverse(&mut self) {
        *self = self.reversed();
    }
}