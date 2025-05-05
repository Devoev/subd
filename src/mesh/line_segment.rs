use crate::mesh::cell::{CellBoundaryTopo, CellTopo, OrderedCellTopo};
use crate::mesh::chain::ChainTopo;
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::vertex::VertexTopo;
use nalgebra::{Const, DimName, DimNameSub, Point2, RealField, U0, U1};
use std::cmp::minmax;

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
pub struct LineSegmentTopo(pub [VertexTopo; 2]);

impl LineSegmentTopo {

    /// Returns the start node of this edge.
    pub fn start(&self) -> VertexTopo {
        self.0[0]
    }

    /// Returns the end node of this edge.
    pub fn end(&self) -> VertexTopo {
        self.0[1]
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

impl CellTopo<U1> for LineSegmentTopo {
    fn nodes(&self) -> &[VertexTopo] {
        &self.0
    }

    fn connected_to<const M: usize>(&self, other: &Self) -> bool
    where
        U1: DimNameSub<Const<M>>
    {
        match M {
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
            _ => unreachable!("Dimension `M` (is {M}) should be <= `K` (is 1)")
        }
    }
}

impl CellBoundaryTopo<U1> for LineSegmentTopo {
    type BoundaryCell = VertexTopo;
    type Boundary = LineSegmentBndTopo;

    fn boundary(&self) -> Self::Boundary {
        LineSegmentBndTopo(self.0)
    }
}

impl OrderedCellTopo<U1> for LineSegmentTopo {
    fn sorted(&self) -> Self {
        LineSegmentTopo(minmax(self.start(), self.end()))
    }
}

pub struct LineSegmentBndTopo(pub [VertexTopo; 2]);

impl ChainTopo<U0, VertexTopo> for LineSegmentBndTopo {
    fn cells(&self) -> &[VertexTopo] {
        &self.0
    }
}