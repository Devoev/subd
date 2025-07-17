use nalgebra::{center, DimName, DimNameSub, Point, RealField, U2};
use crate::cells::line_segment::{LineSegment, NodePair};
use crate::cells::node::NodeIdx;
use crate::cells::quad::{Quad, QuadTopo};
use crate::cells::topo;

// todo: does it make sense, to define new elements?
//  implement Stencils instead, that automatically do the refinement and modify properties of LinSubd

/// A linearly-subdivided quad with subdivision stencil
/// ```text
/// 1/4 --- 1/4
///  |       |
///  |   ○   |
///  |       |
/// 1/4 --- 1/4
/// ```
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct LinFace(pub QuadTopo);

impl LinFace {
    /// Returns the nodes as in [`QuadTopo::nodes`].
    pub fn nodes(&self) -> [NodeIdx; 4] {
        self.0.0
    }

    /// Returns the edges as in [`QuadTopo::edges`] as `LinEdge`s.
    pub fn edges(&self) -> [LinEdge; 4] {
        self.0.edges().map(LinEdge)
    }

    /// Linearly subdivides the given `quad` coordinates.
    pub fn refine<T: RealField, const M: usize>(quad: &Quad<T, M>) -> Point<T, M> {
        quad.centroid()
    }
}

impl topo::Cell<U2> for LinFace {
    fn nodes(&self) -> &[NodeIdx] {
        &self.0.0
    }

    fn is_connected<M: DimName>(&self, other: &Self, dim: M) -> bool
    where
        U2: DimNameSub<M>
    {
        self.0.is_connected(&other.0, dim)
    }
}

/// A linearly-subdivided edge with subdivision stencil
/// ```text
/// 1/2 --- ○ --- 1/2
/// ```
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct LinEdge(pub(crate) NodePair);

impl LinEdge {
    /// Linearly subdivision the given `line` coordinates.
    pub fn refine<T: RealField, const M: usize>(line: &LineSegment<T, M>) -> Point<T, M> {
        let [a, b] = &line.vertices;
        center(a, b)
    }
}