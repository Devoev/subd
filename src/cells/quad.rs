use crate::mesh::elem_vertex::QuadVertexMesh;
use nalgebra::{Const, DimNameSub, Point2, RealField, Vector2, U1, U2};
use crate::cells::topo::{Cell, CellBoundary};
use crate::cells::chain::{Chain, ChainBoundary};
use crate::cells::line_segment::LineSegmentTopo;
use crate::cells::node::NodeIdx;

/// A 2d quadrilateral element of topology [`QuadTopo`].
pub struct Quad<T: RealField> {
    pub vertices: [Point2<T>; 4]
}

impl<T: RealField> Quad<T> {

    /// Constructs a new [`Quad`] from the given `vertices`.
    pub fn new(vertices: [Point2<T>; 4]) -> Self {
        Quad { vertices }
    }

    /// Constructs a new [`Quad`] from the given `topology` and `msh`.
    pub fn from_msh(topology: QuadTopo, msh: &QuadVertexMesh<T>) -> Self {
        Quad::new(topology.0.map(|n| msh.coords(n).clone()))
    }

    /// Computes the centroid of this face.
    pub fn centroid(&self) -> Point2<T> {
        let centroid = self.vertices
            .iter()
            .map(|p| &p.coords)
            .sum::<Vector2<T>>() / T::from_f64(4.0).unwrap();
        Point2::from(centroid)
    }
}


/// Topology of a 2D quadrilateral. The topology is defined as
/// ```text
/// v 3 --- 2
/// ^ |     |
/// | 0 --- 1
/// +---> u
/// ```
/// where `0,1,2,3` are the corner nodes of the quadrilateral.
#[derive(Debug, Clone, Copy)]
pub struct QuadTopo(pub [NodeIdx; 4]);

impl QuadTopo {
    /// Returns the corner nodes.
    pub fn nodes(&self) -> [NodeIdx; 4] {
        self.0
    }

    /// Returns all 4 edges of this quadrilateral face in the following order
    /// ```text
    ///   + -- 2 -- +
    ///   |         |
    /// v 3         1
    /// ^ |         |
    /// | + -- 0 -- +
    /// +---> u
    /// ```
    pub fn edges(&self) -> [LineSegmentTopo; 4] {
        let [a, b, c, d] = self.0;
        [LineSegmentTopo([a, b]), LineSegmentTopo([b, c]), LineSegmentTopo([c, d]), LineSegmentTopo([d, a])]
    }

    // todo: return an intersection result (possibly an enum)
    /// Returns the intersection between `self` and `other` as an iterator of the overlapping nodes.
    pub fn intersection(&self, other: QuadTopo) -> impl Iterator<Item=NodeIdx> {
        self.nodes().into_iter().filter(move |n| other.nodes().contains(n))
    }

    /// Returns whether `self` and `other` are adjacent, i.e. share an edge.
    pub fn is_adjacent(&self, other: QuadTopo) -> bool {
        self.intersection(other).count() == 2
    }

    /// Returns whether `self` and `other` are touching, i.e. share an edge or a node.
    pub fn is_touching(&self, other: QuadTopo) -> bool {
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
    pub fn sorted_by_node(&self, node: NodeIdx, local_idx: usize) -> QuadTopo {
        let original_idx = self.nodes().iter().position(|&n| n == node).unwrap();
        let mut nodes = self.nodes();
        if local_idx > original_idx {
            nodes.rotate_right(local_idx - original_idx);
        } else {
            nodes.rotate_left(original_idx - local_idx);
        }
        QuadTopo(nodes)
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
    pub fn sorted_by_origin(&self, uv_origin: NodeIdx) -> QuadTopo {
        self.sorted_by_node(uv_origin, 0)
    }
}

impl Cell<U2> for QuadTopo {
    fn nodes(&self) -> &[NodeIdx] {
        &self.0
    }

    fn is_connected<const M: usize>(&self, other: &Self) -> bool
    where
        U2: DimNameSub<Const<M>>
    {
        let num_shared_nodes = self.nodes()
            .into_iter()
            .filter(|n| other.nodes().contains(n))
            .count();
        
        match M { 
            2 => num_shared_nodes == 4, // faces are the same
            1 => num_shared_nodes == 2, // faces share an edge
            0 => num_shared_nodes == 1, // faces share a node
            _ => unreachable!("Dimension `M` (is {M}) should be <= `K` (is 2)"),
        }
    }
}

impl CellBoundary<U2> for QuadTopo {
    const NUM_SUB_CELLS: usize = 4;
    type SubCell = LineSegmentTopo;
    type Boundary = QuadBndTopo;

    fn boundary(&self) -> Self::Boundary {
        QuadBndTopo(self.edges())
    }
}

pub struct QuadBndTopo(pub [LineSegmentTopo; 4]);

impl Chain<U1, LineSegmentTopo> for QuadBndTopo {
    fn cells(&self) -> &[LineSegmentTopo] {
        &self.0
    }
}

impl ChainBoundary<U1, LineSegmentTopo> for QuadBndTopo {
    type Boundary = ();

    fn boundary(&self) -> Self::Boundary { }
}